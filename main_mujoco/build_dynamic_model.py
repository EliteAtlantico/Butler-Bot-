#!/usr/bin/env python3
"""Turn the visual-only chopped_urdf_v2 MJCF into a simulatable BracketBot.

The URDF that ships with the model describes a statue: the base is one long
chain of rigidly-welded cover parts, the wheels are welded to the head, there is
no collision geometry anywhere, and the Onshape export left every link with a
volume-ish "mass" (0.29 kg for the whole robot).

This script rebuilds it:

  * the whole static base is flattened into one `chassis` body with a freejoint
  * the two wheels are lifted out and re-parented to the chassis on hinges
  * collision primitives are added (wheels, battery box, mast, head, forearms)
  * masses are re-distributed to a realistic budget, preserving the CAD's
    relative distribution within each group
  * cameras are added: head stereo pair, head depth (RealSense), two wrist cams
  * velocity actuators on the wheels (ODrive velocity mode), position actuators
    on the 18 arm joints, and an IMU/encoder sensor suite

Run:  python build_dynamic_model.py
Out:  chopped_dynamic.xml  (+ scene_dynamic.xml is hand-written alongside)
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "chopped_urdf_v2.xml"
DST = HERE / "chopped_dynamic.xml"

# ---------------------------------------------------------------- mass budget
# The real robot, from BracketBotCapstone/quickstart lib/lqr.py:
#   wheel 2.2 kg each, chassis 3.38 kg.  The arms are v1 hardware the LQR
#   parameters predate, so 1.2 kg/arm is an estimate.
WHEEL_MASS = 2.2          # kg, per wheel (tire + cap), from lqr.py `Mr`
WHEEL_INERTIA_SPIN = 0.018  # kg m^2, from lqr.py `Jr`
CHASSIS_MASS = 3.38       # kg, from lqr.py `Mp` = 4 - 0.62
ARM_MASS = 1.2            # kg, per arm (shoulder through fingers)

WHEEL_RADIUS = 0.0846     # m, measured from the tire mesh; matches lqr.py `R`
WHEEL_HALFWIDTH = 0.0226  # m, measured from the tire mesh
AXLE_Z = 0.0846
AXLE_Y = 0.1611           # tire centre; half-track

ARM_ROOT = "arm_base"
WHEEL_BODIES = {
    "right": ["right_wheel_tire__right_wheel_tire", "right_wheel_cap__right_wheel_cap"],
    "left": ["left_wheel_tire__left_wheel_tire", "left_wheel_cap__left_wheel_cap"],
}
WHEEL_SIGN = {"right": -1.0, "left": +1.0}


def mat2quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).flatten())
    return q


def fmt(v, prec=6):
    return " ".join(f"{float(x):.{prec}g}" for x in np.atleast_1d(v))


def mesh_geom_xml_pose(model, geom_id, want_pos, want_quat):
    """XML pos/quat for a mesh geom that lands it at a given world pose.

    MuJoCo's compiler recentres every mesh's vertices on the mesh frame and
    folds that frame into the compiled geom pose, so `data.geom_xpos` already
    includes it. Writing that value straight back out as the new geom's `pos`
    makes the next compile apply the mesh frame a SECOND time -- which is what
    turns a robot into a pile of parts scattered across the floor. Undo the
    mesh frame here so the round trip is identity.
    """
    mid = model.geom_dataid[geom_id]
    mpos = model.mesh_pos[mid]
    mquat = model.mesh_quat[mid]
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, np.asarray(mquat, float))
    q_xml = np.zeros(4)
    mujoco.mju_mulQuat(q_xml, np.asarray(want_quat, float), inv)
    rotated = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated, np.asarray(mpos, float), q_xml)
    return np.asarray(want_pos, float) - rotated, q_xml


def subtree_bodies(model, root_name):
    """Every body id in the subtree rooted at `root_name` (inclusive)."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    out = {root}
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == root:
                out.add(b)
                break
            p = model.body_parentid[p]
    return out


def combine_inertia(model, data, body_ids, scale):
    """Total mass, CoM and inertia-about-CoM (world axes) for a rigid group.

    `scale` multiplies every body's CAD mass, so the CAD's relative mass
    distribution is preserved while the group total becomes physical.
    """
    masses, coms, inertias = [], [], []
    for b in body_ids:
        m = model.body_mass[b] * scale
        if m <= 0:
            continue
        masses.append(m)
        coms.append(data.xipos[b].copy())
        R = data.ximat[b].reshape(3, 3)
        I_local = np.diag(model.body_inertia[b] * scale)
        inertias.append(R @ I_local @ R.T)
    masses = np.array(masses)
    coms = np.array(coms)
    total = masses.sum()
    com = (masses[:, None] * coms).sum(0) / total
    I = np.zeros((3, 3))
    for m, c, Ib in zip(masses, coms, inertias):
        r = c - com
        I += Ib + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return total, com, I


def inertia_to_mjcf(I):
    """Diagonalise an inertia tensor into MJCF (quat, diaginertia)."""
    evals, evecs = np.linalg.eigh(I)
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] *= -1
    return mat2quat(evecs), np.clip(evals, 1e-8, None)


def camera_xyaxes(forward, up):
    """MJCF `xyaxes` for a camera looking along `forward` with `up` upward.

    A MuJoCo camera looks down its own -z with +y up, so x_cam = forward x up.
    """
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, float)
    u -= f * np.dot(u, f)
    u /= np.linalg.norm(u)
    x = np.cross(f, u)
    return np.concatenate([x, u])


def world_to_local(data, body_id, pos_w, axes_w):
    """Express a world-frame camera pose in a body's local frame."""
    R = data.xmat[body_id].reshape(3, 3)
    p = data.xpos[body_id]
    pos_l = R.T @ (np.asarray(pos_w) - p)
    x_l = R.T @ axes_w[:3]
    y_l = R.T @ axes_w[3:]
    return pos_l, np.concatenate([x_l, y_l])


def build():
    model = mujoco.MjModel.from_xml_path(str(SRC))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    src_tree = ET.parse(SRC)
    src_root = src_tree.getroot()

    # ---- partition the bodies -------------------------------------------
    arm_ids = subtree_bodies(model, ARM_ROOT)
    wheel_ids = {
        side: {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
        for side, names in WHEEL_BODIES.items()
    }
    all_wheel_ids = set().union(*wheel_ids.values())
    chassis_ids = {
        b for b in range(1, model.nbody) if b not in arm_ids and b not in all_wheel_ids
    }

    # ---- mass scaling factors -------------------------------------------
    chassis_cad = sum(model.body_mass[b] for b in chassis_ids)
    arm_cad = sum(model.body_mass[b] for b in arm_ids)
    chassis_scale = CHASSIS_MASS / chassis_cad
    arm_scale = (2 * ARM_MASS) / arm_cad

    c_mass, c_com, c_I = combine_inertia(model, data, chassis_ids, chassis_scale)
    c_quat, c_diag = inertia_to_mjcf(c_I)

    print(f"chassis: {c_mass:.3f} kg  CoM={np.round(c_com, 4)}  "
          f"diaginertia={np.round(c_diag, 4)}")

    # ---- build the new document -----------------------------------------
    root = ET.Element("mujoco", model="bracketbot_chopped_v2")
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes/",
                  autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast",
                  cone="elliptic", impratio="10")

    # keep the original asset block verbatim (all 50 meshes)
    root.append(copy.deepcopy(src_root.find("asset")))

    dflt = ET.SubElement(root, "default")
    vis = ET.SubElement(dflt, "default", {"class": "visual"})
    ET.SubElement(vis, "geom", contype="0", conaffinity="0", group="2",
                  type="mesh", density="0")
    col = ET.SubElement(dflt, "default", {"class": "collision"})
    ET.SubElement(col, "geom", contype="1", conaffinity="1", group="3",
                  density="0", rgba="0.9 0.4 0.2 0.35", friction="1 0.005 0.0001")
    whl = ET.SubElement(dflt, "default", {"class": "wheel"})
    ET.SubElement(whl, "geom", contype="1", conaffinity="1", group="3",
                  density="0", rgba="0.1 0.1 0.1 0.6",
                  friction="1.6 0.01 0.001", condim="4",
                  solref="0.005 1", priority="2")
    pad = ET.SubElement(dflt, "default", {"class": "pad"})
    ET.SubElement(pad, "geom", contype="1", conaffinity="1", group="3",
                  density="0", rgba="0.2 0.8 0.4 0.6",
                  friction="2.5 0.05 0.005", condim="4", priority="3",
                  solref="0.004 1", solimp="0.95 0.99 0.001")
    # Arm servo gains are sized to actually HOLD the arm, not just suggest a
    # position to it. At kp=150 the mast sagged 0.27 m below its command under
    # the arm's own 2.4 kg and the grasp site sat 0.4 m below where IK thought
    # it was, which no amount of closed-loop IK can recover from.
    arm = ET.SubElement(dflt, "default", {"class": "arm"})
    ET.SubElement(arm, "joint", damping="4.0", armature="0.05", frictionloss="0.1")
    ET.SubElement(arm, "position", kp="800", kv="60", forcerange="-150 150")
    # The mast is prismatic and carries the entire arm, so it needs stiffness in
    # N/m, not Nm/rad.
    mast = ET.SubElement(dflt, "default", {"class": "mast"})
    ET.SubElement(mast, "joint", damping="60", armature="0.5", frictionloss="1.0")
    ET.SubElement(mast, "position", kp="6000", kv="400", forcerange="-400 400")
    # The gripper needs its own, much softer servo. On the arm class a 7 deg
    # squeeze command turns into ~10 Nm at the finger hinge, which is ~100 N at
    # the pad -- enough to fire a 45 mm cube across the room instead of holding
    # it.
    grip = ET.SubElement(dflt, "default", {"class": "gripper"})
    ET.SubElement(grip, "joint", damping="0.4", armature="0.01",
                  frictionloss="0.02")
    ET.SubElement(grip, "position", kp="35", kv="1.5", forcerange="-12 12")

    wb = ET.SubElement(root, "worldbody")
    chassis = ET.SubElement(wb, "body", name="chassis", pos="0 0 0")
    ET.SubElement(chassis, "freejoint", name="root")
    ET.SubElement(chassis, "inertial", pos=fmt(c_com), quat=fmt(c_quat),
                  mass=f"{c_mass:.6g}", diaginertia=fmt(c_diag))
    ET.SubElement(chassis, "site", name="imu", pos=fmt(c_com), size="0.01",
                  rgba="0 1 0 0.5")

    # ---- helper: AABB of a body's meshes, in a chosen frame --------------
    def mesh_aabb(body_ids, frame_body=0):
        """AABB of every mesh geom on `body_ids`, expressed in `frame_body`."""
        Rf = data.xmat[frame_body].reshape(3, 3)
        pf = data.xpos[frame_body]
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for g in range(model.ngeom):
            if model.geom_bodyid[g] not in body_ids:
                continue
            mid = model.geom_dataid[g]
            if mid < 0:
                continue
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            V = model.mesh_vert[va:va + vn].reshape(-1, 3)
            W = V @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            L = (W - pf) @ Rf
            lo = np.minimum(lo, L.min(0))
            hi = np.maximum(hi, L.max(0))
        return (lo + hi) / 2, (hi - lo) / 2

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    # ---- flatten every static chassis mesh onto the chassis body ---------
    n_vis = 0
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if b not in chassis_ids:
            continue
        mid = model.geom_dataid[g]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        gp, gq = mesh_geom_xml_pose(model, g, data.geom_xpos[g],
                                    mat2quat(data.geom_xmat[g]))
        ET.SubElement(chassis, "geom", {
            "class": "visual", "mesh": mesh_name,
            "pos": fmt(gp), "quat": fmt(gq),
            "rgba": fmt(model.geom_rgba[g], 4),
        })
        n_vis += 1

    # ---- chassis collision primitives -----------------------------------
    low_ids = {b for b in chassis_ids if data.xipos[b][2] < 0.30}
    mast_ids = {bid("main_extrusion__main_extrusion__main_extrusion__main_extrusion")}
    head_ids = {b for b in chassis_ids
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                .startswith(("head", "camera_cover", "head_cover"))}

    for name, ids in (("base_hull", low_ids), ("mast", mast_ids), ("head", head_ids)):
        if not ids:
            continue
        c, h = mesh_aabb(ids)
        if not np.all(np.isfinite(c)):
            continue
        ET.SubElement(chassis, "geom", {
            "class": "collision", "name": f"col_{name}", "type": "box",
            "pos": fmt(c), "size": fmt(np.maximum(h, 1e-3)),
        })
        print(f"  collision {name:10s} centre={np.round(c, 3)} half={np.round(h, 3)}")

    # ---- head cameras ----------------------------------------------------
    head_c, head_h = mesh_aabb(head_ids)
    cam_x = head_c[0] + head_h[0] + 0.005   # just in front of the head shell
    cam_z = head_c[2] + 0.02
    # The head sits 1.5 m up. A camera mounted dead level there looks straight
    # over anything shorter than a coffee table, so the depth camera gets the
    # downward mount tilt a navigation RealSense actually gets. The RGB and
    # stereo cameras stay level -- those are for looking at the world, not for
    # not running into it.
    for cname, dy, fovy, tilt_deg in (("head_rgb", 0.0, 42, 0.0),
                                      ("head_depth", 0.0, 58, 22.0),
                                      ("head_stereo_left", 0.03, 55, 0.0),
                                      ("head_stereo_right", -0.03, 55, 0.0)):
        tilt = np.deg2rad(tilt_deg)
        fwd = camera_xyaxes([np.cos(tilt), 0, -np.sin(tilt)], [0, 0, 1])
        ET.SubElement(chassis, "camera", {
            "name": cname, "mode": "fixed", "fovy": str(fovy),
            "pos": fmt([cam_x, dy, cam_z]), "xyaxes": fmt(fwd),
        })
    print(f"  head cameras at x={cam_x:.3f} z={cam_z:.3f}")

    # ---- wheels: lift out of the static chain onto hinges ----------------
    for side, ids in wheel_ids.items():
        sy = WHEEL_SIGN[side] * AXLE_Y
        wbody = ET.SubElement(chassis, "body", name=f"{side}_wheel",
                              pos=fmt([0.0, sy, AXLE_Z]))
        ET.SubElement(wbody, "joint", {
            "name": f"{side}_wheel_joint", "type": "hinge", "axis": "0 1 0",
            "damping": "0.02", "armature": "0.005", "frictionloss": "0.01",
        })
        # thin disc about its spin (y) axis
        ET.SubElement(wbody, "inertial", pos="0 0 0", mass=f"{WHEEL_MASS:g}",
                      diaginertia=fmt([WHEEL_INERTIA_SPIN / 2,
                                       WHEEL_INERTIA_SPIN,
                                       WHEEL_INERTIA_SPIN / 2]))
        for g in range(model.ngeom):
            if model.geom_bodyid[g] not in ids:
                continue
            mid = model.geom_dataid[g]
            gp, gq = mesh_geom_xml_pose(
                model, g, data.geom_xpos[g] - np.array([0.0, sy, AXLE_Z]),
                mat2quat(data.geom_xmat[g]))
            ET.SubElement(wbody, "geom", {
                "class": "visual",
                "mesh": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid),
                "pos": fmt(gp), "quat": fmt(gq),
                "rgba": fmt(model.geom_rgba[g], 4),
            })
        ET.SubElement(wbody, "geom", {
            "class": "wheel", "name": f"{side}_tire", "type": "cylinder",
            "size": f"{WHEEL_RADIUS:g} {WHEEL_HALFWIDTH:g}",
            "zaxis": "0 1 0",
        })
        ET.SubElement(wbody, "site", name=f"{side}_wheel_site", size="0.005",
                      rgba="1 0 0 0.5")

    # ---- arms: re-parent the subtree verbatim, rescaled ------------------
    arm_el = copy.deepcopy(src_root.find(f".//body[@name='{ARM_ROOT}']"))
    arm_el.set("pos", fmt(data.xpos[bid(ARM_ROOT)]))
    arm_el.set("quat", fmt(data.xquat[bid(ARM_ROOT)]))

    for inert in arm_el.iter("inertial"):
        inert.set("mass", f"{float(inert.get('mass')) * arm_scale:.6g}")
        di = np.array([float(x) for x in inert.get("diaginertia").split()])
        inert.set("diaginertia", fmt(di * arm_scale))
    MAST_JOINTS = {"rj0", "lj0"}
    for jnt in arm_el.iter("joint"):
        n = jnt.get("name")
        jnt.set("class", "gripper" if "gripper" in n
                else "mast" if n in MAST_JOINTS else "arm")
        # The URDF carries <limit effort="10"> on every arm joint, which the
        # importer turns into actuatorfrcrange="-10 10". That is a per-JOINT cap
        # applied on top of whatever the actuator asks for, and it is below the
        # 11.8 N the mast needs just to hold the arm against gravity -- so the
        # arm sinks at constant velocity with its actuator pinned at the limit,
        # no matter how the servo is tuned. Drop it and let each actuator's own
        # forcerange govern.
        jnt.attrib.pop("actuatorfrcrange", None)

    # collision boxes on the parts that can actually hit something. The fingers
    # are deliberately NOT in this list: an AABB of a whole finger is 126 mm
    # long and the two of them interpenetrate at the closed position, so the
    # gripper spends its torque fighting itself. They get fingertip pads below.
    ARM_COLLIDE = ["forearm__forearm", "l_forearm__forearm", "hand__hand",
                   "l_hand__hand"]
    for bname in ARM_COLLIDE:
        el = arm_el.find(f".//body[@name='{bname}']")
        if el is None:
            continue
        c, h = mesh_aabb({bid(bname)}, frame_body=bid(bname))
        if not np.all(np.isfinite(c)):
            continue
        ET.SubElement(el, "geom", {
            "class": "collision", "name": f"col_{bname}", "type": "box",
            "pos": fmt(c), "size": fmt(np.maximum(h, 2e-3)),
        })


    # ---- fingertip pads + grasp sites -----------------------------------
    # Small world-axis-aligned pads at the measured fingertips, thin enough in
    # the gripping direction to leave a gap when the gripper is closed, with
    # high friction so a grasp holds by friction rather than by a cheat weld.
    FINGERS = {
        "right": ("left_finger__left_finger", "right_finger__right_finger",
                  "hand__hand"),
        "left": ("l_left_finger__left_finger", "l_right_finger__right_finger",
                 "l_hand__hand"),
    }
    PAD_HALF = np.array([0.020, 0.005, 0.018])   # x fwd, y grip, z along finger

    def fingertip_world(body_name, depth=0.012):
        """Centroid of the last `depth` metres of a finger, in world coords."""
        b = bid(body_name)
        pts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b or model.geom_dataid[g] < 0:
                continue
            mid = model.geom_dataid[g]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            V = model.mesh_vert[va:va + vn].reshape(-1, 3)
            pts.append(V @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
        P = np.vstack(pts)
        return P[P[:, 2] < P[:, 2].min() + depth].mean(0)

    for side, (fa, fb, hand) in FINGERS.items():
        tips = []
        for fname in (fa, fb):
            tip = fingertip_world(fname)
            tips.append(tip)
            b = bid(fname)
            R = data.xmat[b].reshape(3, 3)
            el = arm_el.find(f".//body[@name='{fname}']")
            inv_quat = np.zeros(4)
            mujoco.mju_mat2Quat(inv_quat, R.T.flatten())
            ET.SubElement(el, "geom", {
                "class": "pad", "name": f"pad_{fname}", "type": "box",
                "pos": fmt(R.T @ (tip - data.xpos[b])),
                "quat": fmt(inv_quat), "size": fmt(PAD_HALF),
            })
        # grasp site: midway between the fingertips, on the hand body
        mid_w = (tips[0] + tips[1]) / 2
        hb = bid(hand)
        Rh = data.xmat[hb].reshape(3, 3)
        inv_h = np.zeros(4)
        mujoco.mju_mat2Quat(inv_h, Rh.T.flatten())
        hand_el = arm_el.find(f".//body[@name='{hand}']")
        ET.SubElement(hand_el, "site", {
            "name": f"{side}_grasp", "pos": fmt(Rh.T @ (mid_w - data.xpos[hb])),
            "quat": fmt(inv_h), "size": "0.008",
            "rgba": "0 1 0.3 0.6", "group": "4",
        })
        print(f"  {side} grasp site at {np.round(mid_w, 4)}  "
              f"fingertip gap {np.linalg.norm(tips[0] - tips[1]) * 1000:.1f} mm")

    # wrist cameras, aimed from the camera mount toward that arm's end effector
    for side, cam_body, eef in (("right", "wrist_cam__wrist_cam", "right_eef"),
                                ("left", "l_wrist_cam__wrist_cam", "left_eef")):
        cb, eb = bid(cam_body), bid(eef)
        look = data.xpos[eb] - data.xpos[cb]
        axes_w = camera_xyaxes(look, [0, 0, 1])
        pos_l, axes_l = world_to_local(data, cb, data.xpos[cb], axes_w)
        el = arm_el.find(f".//body[@name='{cam_body}']")
        ET.SubElement(el, "camera", {
            "name": f"wrist_cam_{side}", "mode": "fixed", "fovy": "60",
            "pos": fmt(pos_l), "xyaxes": fmt(axes_l),
        })
    chassis.append(arm_el)

    ET.SubElement(chassis, "camera", {
        "name": "chase", "mode": "trackcom", "pos": "-2.2 -2.2 1.6",
        "xyaxes": "0.707 -0.707 0 0.29 0.29 0.91",
    })

    # ---- actuators -------------------------------------------------------
    # Wheels are TORQUE actuators on purpose: the real ODrive closes its own
    # velocity loop onto current, so ODriveSim does the same in Python and both
    # the LQR (torque) and set_speed_mps (velocity) paths share one actuator
    # instead of fighting each other.
    act = ET.SubElement(root, "actuator")
    for side in ("left", "right"):
        ET.SubElement(act, "motor", {
            "name": f"{side}_wheel", "joint": f"{side}_wheel_joint",
            "gear": "1", "ctrlrange": "-15 15",
        })
    arm_joints = [j.get("name") for j in arm_el.iter("joint")]
    for jn in arm_joints:
        ET.SubElement(act, "position", {
            "class": ("gripper" if "gripper" in jn
                      else "mast" if jn in MAST_JOINTS else "arm"),
            "name": f"act_{jn}", "joint": jn,
        })

    # ---- sensors ---------------------------------------------------------
    sen = ET.SubElement(root, "sensor")
    ET.SubElement(sen, "framequat", name="imu_quat", objtype="site", objname="imu")
    ET.SubElement(sen, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sen, "accelerometer", name="imu_acc", site="imu")
    ET.SubElement(sen, "framepos", name="base_pos", objtype="site", objname="imu")
    ET.SubElement(sen, "framelinvel", name="base_linvel", objtype="site", objname="imu")
    for side in ("left", "right"):
        ET.SubElement(sen, "jointpos", name=f"{side}_wheel_pos",
                      joint=f"{side}_wheel_joint")
        ET.SubElement(sen, "jointvel", name=f"{side}_wheel_vel",
                      joint=f"{side}_wheel_joint")

    # gripper mimic constraints, carried over from the URDF
    eq = src_root.find("equality")
    if eq is not None:
        root.append(copy.deepcopy(eq))

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(DST, encoding="unicode", xml_declaration=False)
    print(f"\nwrote {DST}  ({n_vis} visual meshes flattened, "
          f"{len(arm_joints)} arm joints)")

    # ---- verify ----------------------------------------------------------
    m2 = mujoco.MjModel.from_xml_path(str(DST))
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    total = m2.body_mass.sum()
    com = (m2.body_mass[:, None] * d2.xipos).sum(0) / total
    print(f"verified: nq={m2.nq} nv={m2.nv} nu={m2.nu} ngeom={m2.ngeom} "
          f"ncam={m2.ncam} nsensor={m2.nsensor}")
    print(f"          total mass {total:.3f} kg   CoM {np.round(com, 4)}   "
          f"CoM height above axle {com[2] - AXLE_Z:.4f} m")
    return m2


if __name__ == "__main__":
    build()
