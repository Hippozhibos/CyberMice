import xml.etree.ElementTree as ET
import math

# Parse the XML file
tree = ET.parse(r'D:\Mujoco\CyberMice\assets\CyberMiceJointActuated.xml')
root = tree.getroot()

# Find the 'CyberMice' body
cybermice_body = root.find(".//body[@name='CyberMice']")

# Update rotation values from degrees to radians for geom elements
for geom in cybermice_body.findall('.//geom'):
    geom_euler = geom.attrib.get('euler')
    if geom_euler:
        rotation_values = geom_euler.split(' ')
        rotation_values_in_radians = [str(math.radians(float(angle))) for angle in rotation_values]
        geom.attrib['euler'] = ' '.join(rotation_values_in_radians)

# Update rotation values from degrees to radians for joint elements
for joint in cybermice_body.findall('.//joint'):
    joint_range = joint.attrib.get('range')
    if joint_range:
        range_values = joint_range.split(' ')
        range_values_in_radians = [str(math.radians(float(angle))) for angle in range_values]
        joint.attrib['range'] = ' '.join(range_values_in_radians)

# Save the modified XML file
tree.write('CyberMiceJointActuated_2.xml')

# Now load the modified XML file into dm_control
# Your dm_control loading code goes here
