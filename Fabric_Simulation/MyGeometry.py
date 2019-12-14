import pychrono as chrono
from pychrono import irrlicht as chronoirr
def setSys(gz):

    ## 1. Create the physical system that will handle all bodies and constraints.
    ##    global reference frame having Z up.
    system = chrono.ChSystemNSC()
    ##    Specify the gravitational acceleration vector, consistent with the
    system.Set_G_acc(chrono.ChVectorD(gz[0], gz[1], gz[2]))
    return system

## 2. Create the rigid bodies of the slider-crank mechanical system.
##    For each body, specify:
##    - a unique identifier
##    - mass and moments of inertia
##    - position and orientation of the (centroidal) body frame
##    - visualization assets (defined with respect to the body frame)

def setCrank(identifier, pos, Orient, length,system):
    crank = chrono.ChBody()
    crank.SetIdentifier(identifier)
    crank.SetName("crank" + str(identifier))
    crank.SetMass(1.0)
    crank.SetInertiaXX(chrono.ChVectorD(0.005, 0.1, 0.1))

    crank.SetPos(chrono.ChVectorD(pos[0], pos[1], pos[2]))
    crank.SetRot(Orient)

    box_c = chrono.ChBoxShape()
    box_c.GetBoxGeometry().Size = chrono.ChVectorD(length, 0.05, 0.05)
    crank.AddAsset(box_c)

    col_c = chrono.ChColorAsset()
    col_c.SetColor(chrono.ChColor(0.6, 0.2, 0.2))
    crank.AddAsset(col_c)
    system.AddBody(crank)
    return crank


def setGround(system):
    ## Ground
    ground = chrono.ChBody()
    ground.SetIdentifier(-1)
    ground.SetName("ground")
    ground.SetBodyFixed(True)

    cyl_g = chrono.ChCylinderShape()
    cyl_g.GetCylinderGeometry().p1 = chrono.ChVectorD(-10, 0.2+2.5, -2.5)
    cyl_g.GetCylinderGeometry().p2 = chrono.ChVectorD(-10, -0.2+2.5, -2.5)
    cyl_g.GetCylinderGeometry().rad = 0.03
    ground.AddAsset(cyl_g)

    col_g = chrono.ChColorAsset()
    col_g.SetColor(chrono.ChColor(0.6, 0.6, 0.2))
    ground.AddAsset(col_g)
    system.AddBody(ground)

    return ground


def addRevoluteLink(ground, crank1, system):
    ## Define two quaternions representing:
    ## - a rotation of -90 degrees around x (z2y)
    ## - a rotation of +90 degrees around y (z2x)
    z2y = chrono.ChQuaternionD()
    z2x = chrono.ChQuaternionD()
    z2y.Q_from_AngAxis(-chrono.CH_C_PI / 2, chrono.ChVectorD(1, 0, 0))
    z2x.Q_from_AngAxis(chrono.CH_C_PI / 2, chrono.ChVectorD(0, 1, 0))

    ## revolute joint between ground and crank. As before, we apply the 'z2y'
    ## rotation to align the rotation axis with the Y axis of the global frame.
    engine_ground_crank = chrono.ChLinkRevolute()
    engine_ground_crank.SetName("engine_ground_crank")
    engine_ground_crank.Initialize(ground, crank1, chrono.ChFrameD(chrono.ChVectorD(0, 0, 0), z2y))
    system.AddLink(engine_ground_crank)
    return engine_ground_crank

def addSphericalJoint(crank1, crank2, pCons, system,addDamping):

    ## Spherical joint between ground and crank.
    spherical_crank_rod = chrono.ChLinkLockSpherical()
    spherical_crank_rod.SetName("spherical_crank_rod")
    spherical_crank_rod.Initialize(crank1, crank2, chrono.ChCoordsysD(chrono.ChVectorD(pCons[0], pCons[1], pCons[2]), chrono.QUNIT))
    system.AddLink(spherical_crank_rod)
    if addDamping:
        addSpringDampners(crank1, crank2, system)
    return spherical_crank_rod

def addSpringDampners(crank1, crank2,system):
    #### Add a spring-damper (ChLinkspring) between ground and the ball.
    #### This element should connect the center of the ball with the global point
    #### (6.5, 0, 0).  Set a spring constant of 50 and a spring free length of 1.
    #### Set a damping coefficient of 5.
    sp = chrono.ChLinkSpring()
    sp.Set_SpringK(10)
    sp.Set_SpringR(58.0)
    sp.Set_SpringRestLength(5.0)
    sp.Initialize(crank1, crank2, False, crank1.GetPos(), crank2.GetPos())
    # engine_ground_crank.SetSpeedFunction(fun)
    system.AddLink(sp)
    return sp

def addSpringDampnersToGround(crank1, crank2,system):
    #### Add a spring-damper (ChLinkspring) between ground and the ball.
    #### This element should connect the center of the ball with the global point
    #### (6.5, 0, 0).  Set a spring constant of 50 and a spring free length of 1.
    #### Set a damping coefficient of 5.
    sp = chrono.ChLinkSpring()
    sp.Set_SpringK(0)
    sp.Set_SpringR(60.0)
    sp.Set_SpringRestLength(19.0)
    sp.Initialize(crank1, crank2, False, crank1.GetPos(), crank2.GetPos())
    # engine_ground_crank.SetSpeedFunction(fun)
    system.AddLink(sp)
    return sp

## Create the Irrlicht application and set-up the camera.
def setApp(system):
    application = chronoirr.ChIrrApp(
        system,  ## pointer to the mechanical system
        "Slider-Crank Demo 2",  ## title of the Irrlicht window
        chronoirr.dimension2du(800, 600),  ## window dimension (width x height)
        False,  ## use full screen?
        True)  ## enable shadows?
    application.AddTypicalLogo()
    application.AddTypicalSky()
    application.AddTypicalLights()
    application.AddTypicalCamera(
        chronoirr.vector3df(2, 5, -3),  ## camera location
        chronoirr.vector3df(2, 0, 0))  ## "look at" location

    ## Let the Irrlicht application convert the visualization assets.
    application.AssetBindAll()
    application.AssetUpdateAll()

    ## 6. Perform the simulation.

    ## Specify the step-size.
    application.SetTimestep(0.01)
    application.SetTryRealtime(True)
    return application
