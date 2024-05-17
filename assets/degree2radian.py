import xml.etree.ElementTree as ET
import math

# Parse the XML file
tree = ET.parse(r'D:\Mujoco\CyberMice\assets\CyberMice_0517.xml')
root = tree.getroot()

# Find and update rotation values from degrees to radians
for body in root.findall('.//body'):
    print(body.attrib.get('name'))
    for geom in body.findall('.//geom'):
        # geom_name = geom.attrib.get('name')
        # print(geom_name)
        geom_euler = geom.attrib.get('euler')  # Get the 'type' attribute if it exists
        if geom_euler:
            rotation_attrib = geom_euler
            if rotation_attrib:
                print("Original rotation values:", rotation_attrib)  # Print original values for verification
                # Split the rotation values and convert each from degrees to radians
                rotation_values = rotation_attrib.split(' ')
                rotation_values_in_radians = [str(math.radians(float(angle))) for angle in rotation_values]
                print("Rotation values in radians:", rotation_values_in_radians)  # Print radians values for verification
                # Update the rotation attribute with values in radians
                geom.attrib['euler'] = ' '.join(rotation_values_in_radians)

    for joint in body.findall('.//joint'):
        # joint_name = joint.attrib.get('name')
        # print(joint_name)
        joint_range = joint.attrib.get('range')  # Get the 'type' attribute if it exists
        if joint_range:
            range_attrib = joint_range
            if range_attrib:
                print("Original range values:", range_attrib)  # Print original values for verification
                # Split the rotation values and convert each from degrees to radians
                range_values = range_attrib.split(' ')
                range_values_in_radians = [str(math.radians(float(angle))) for angle in range_values]
                print("range values in radians:", range_values_in_radians)  # Print radians values for verification
                # Update the rotation attribute with values in radians
                joint.attrib['range'] = ' '.join(range_values_in_radians)

# Save the modified XML file
tree.write('modified_xml_file_2.xml')

# Now load the modified XML file into dm_control
# Your dm_control loading code goes here
