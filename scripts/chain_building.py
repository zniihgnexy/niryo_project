import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from ikpy.chain import Chain
from ikpy.link import URDFLink
import numpy as np

def parse_mujoco_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return root

def extract_joints_links(mujoco_root):
    joints = []
    links = []

    for body in mujoco_root.findall('.//body'):
        for joint in body.findall('joint'):
            pos = joint.get('pos')
            if pos:
                pos = [float(x) for x in pos.split()]
            else:
                pos = [0.0, 0.0, 0.0]  # Default position if 'pos' is not specified
            
            joint_info = {
                'name': joint.get('name'),
                'type': joint.get('type'),
                'pos': pos,
                'axis': [float(x) for x in joint.get('axis').split()] if joint.get('axis') else [0, 0, 1],  # Default axis if not specified
                'range': [float(x) for x in joint.get('range').split()] if joint.get('range') else None
            }
            joints.append(joint_info)

        for geom in body.findall('geom'):
            link_info = {
                'name': geom.get('name'),
                'type': geom.get('type'),
                'pos': [float(x) for x in geom.get('pos').split()] if geom.get('pos') else [0.0, 0.0, 0.0]  # Default position if 'pos' is not specified
            }
            links.append(link_info)

    return joints, links

def quat_to_euler(quat):
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=False)

def build_ikpy_chain(joints, links):
    ik_chain = []

    for i, joint in enumerate(joints):
        name = joint['name']
        pos = joint['pos']
        axis = joint['axis']
        joint_range = joint['range'] if joint['range'] else (-3.14, 3.14)  # Default range

        # Convert axis to rotation angles
        if 'quat' in joint:
            orientation = quat_to_euler(joint['quat'])
        else:
            orientation = [0, 0, 0]  # Default orientation

        link = URDFLink(
            name=name,
            origin_translation=pos,
            origin_orientation=orientation,
            rotation=axis,
            bounds=joint_range
        )

        ik_chain.append(link)

    return Chain(name='robot', links=ik_chain)

# Parse the MuJoCo XML file
mujoco_xml_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
mujoco_root = parse_mujoco_xml(mujoco_xml_path)

# Extract joints and links
joints, links = extract_joints_links(mujoco_root)

# Build IKPy kinematic chain
ikpy_chain = build_ikpy_chain(joints, links)


breakpoint()
# Perform inverse kinematics
target_position = [0.1, 0.1, 0.1]
target_frame = np.eye(4)
target_frame[:3, 3] = target_position

initial_position = [0] * len(ikpy_chain.links)
ik_solution = ikpy_chain.inverse_kinematics(target_frame, initial_position=initial_position)
print("IK Solution:", ik_solution)
