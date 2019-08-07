from builtins import range
import pybullet as p
import time
import pybullet_data
cin = p.connect(p.SHARED_MEMORY)
if (cin < 0):
    cin = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
objects = [p.loadURDF("plane.urdf", 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000)]
objects = [p.loadURDF("/URDF/ALPHRED/urdf/ALPHRED.urdf", 0.004912,0.004855,0.122558,-0.019988,0.020242,-0.000034,0.999595)]
ob = objects[0]
jointPositions=[ -0.000028, 0.000180, 0.000976, -0.000033, 0.000217, -0.000017, -0.000003, 0.000049, 0.001236, -0.000022, 0.000010, 0.000000 ]
for jointIndex in range (p.getNumJoints(ob)):
	p.resetJointState(ob,jointIndex,jointPositions[jointIndex])

p.setGravity(0.000000,0.000000,-10.000000)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect()
