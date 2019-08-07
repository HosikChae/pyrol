from builtins import range
import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0.8]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("/URDF/ALPHRED/urdf/ALPHRED.urdf",cubeStartPos, cubeStartOrientation)
jointPositions = [ 0.0, 0.7, 0.7, 0.0, 0.7, 0.7, 0.0, 0.7, 0.7, 0.0, -3.14/2.0, 0.0]
for jointIndex in range (p.getNumJoints(boxId)):
    p.resetJointState(boxId,jointIndex,jointPositions[jointIndex])
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
p.disconnect()
