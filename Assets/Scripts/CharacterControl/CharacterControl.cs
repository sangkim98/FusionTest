using UnityEngine;
using System.Collections.Generic;
using System;
using Unity.Sentis;
using UnityEngine.Animations.Rigging;
using DuloGames.UI.Tweens;

public class CharacterControl : MonoBehaviour {

    // Pose Estimator
    WebCamTexture webcamTexture;
    PoseEstimator poseEstimator;
    const int resizedSquareImageDim = 320;

    // Game Objects
    private List<GameObject> targetThreeDPoints;

    // Sentis    
    public ModelAsset twoDPoseModelAsset;
    public ModelAsset threeDPoseModelAsset;

    // IK Control
    // public bool followPose = false;
    // public bool lowerBody = false;
    [SerializeField, Range(0,1)] public float weight = 1.0f;
    private Rig myrig;
    // For joint control
    public Transform characterRoot;
    public Transform belly;
    public Transform neck;
    public Transform nose;
    public Transform leftClavicle;
    public Transform leftShoulder;
    public Transform leftElbow;
    public Transform leftWrist;
    public Transform rightClavicle;
    public Transform rightShoulder;
    public Transform rightElbow;
    public Transform rightWrist;

    private float[] boneDistances;

    void Start() {

        WebCamDevice[] devices = WebCamTexture.devices;
        
        webcamTexture = new WebCamTexture(devices[0].name, 640, 360, 30);

        webcamTexture.Play();

        // Sentis model initialization
        poseEstimator = new PoseEstimator(resizedSquareImageDim, ref twoDPoseModelAsset, ref threeDPoseModelAsset, BackendType.GPUCompute);

        // IK setup

        init3DKeypoints();
        SetupRig();

    }

    void Update() {

        bool goodEstimate;

        goodEstimate = poseEstimator.RunML(webcamTexture);

        if(goodEstimate) {

            Vector3[] threeDJoints = poseEstimator.getThreeDPose();

            Draw3DPoints(threeDJoints);

        }

        myrig.weight = weight;

    }
    private void init3DKeypoints() {

        targetThreeDPoints = new List<GameObject>();
        GameObject root = new GameObject("pose_root");
        root.transform.SetParent(characterRoot);
        root.transform.localPosition = Vector3.zero;
        root.transform.Rotate(0,0,0);

        for (int i = 0; i < 17; i++) {

            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            sphere.name = String.Format("joint{0}", i);

            sphere.transform.localScale = new Vector3(0.05f,0.05f,0.05f);
            sphere.transform.localPosition = new Vector3(0,0,0);

            sphere.transform.SetParent(root.transform, false);

            targetThreeDPoints.Add(sphere);

        }

    }

    void Draw3DPoints(Vector3[] joints) {

        Vector3 rootToBelly = fromAtoB(joints[0], joints[7]);
        Vector3 bellyToNeck = fromAtoB(joints[7], joints[8]);
        Vector3 neckToNose = fromAtoB(joints[8], joints[9]);
        Vector3 neckToLeftShoulder = fromAtoB(joints[8], joints[11]);
        Vector3 leftShoulderToElbow = fromAtoB(joints[11], joints[12]);
        Vector3 leftElbowToWrist = fromAtoB(joints[12], joints[13]);
        Vector3 neckToRightShoulder = fromAtoB(joints[8], joints[14]);
        Vector3 rightShoulderToElbow = fromAtoB(joints[14], joints[15]);
        Vector3 rightElbowToWrist = fromAtoB(joints[15], joints[16]);

        rootToBelly.Normalize();
        bellyToNeck.Normalize();
        neckToNose.Normalize();
        neckToLeftShoulder.Normalize();
        leftShoulderToElbow.Normalize();
        leftElbowToWrist.Normalize();
        neckToRightShoulder.Normalize();
        rightShoulderToElbow.Normalize();
        rightElbowToWrist.Normalize();

        joints[7] = joints[0] + rootToBelly * boneDistances[0];
        joints[8] = joints[7] + bellyToNeck * boneDistances[1];
        joints[9] = joints[8] + neckToNose * boneDistances[2];
        joints[11] = joints[8] + neckToLeftShoulder * boneDistances[6];
        joints[12] = joints[11] + leftShoulderToElbow * boneDistances[7];
        joints[13] = joints[12] + leftElbowToWrist * boneDistances[8];
        joints[14] = joints[8] + neckToRightShoulder * boneDistances[3];
        joints[15] = joints[14] + rightShoulderToElbow * boneDistances[4];
        joints[16] = joints[15] + rightElbowToWrist * boneDistances[5];

        for (int idx = 0; idx < joints.Length; idx++) {

            if(idx == 10) continue;

            GameObject point = targetThreeDPoints[idx];
            point.transform.localPosition = joints[idx];

        }

    }

    void SetupRig() {

        RigBuilder rigBuilder = gameObject.AddComponent<RigBuilder>();
        GameObject rig1 = new GameObject("Rig1");
        myrig = rig1.AddComponent<Rig>();
        myrig.weight = weight;

        rig1.transform.SetParent(gameObject.transform);

        rigBuilder.layers.Add(new RigLayer(rig1.GetComponent<Rig>(), true));

        GameObject lookAt = createMultiAimConstraint("lookAt", rig1.transform);
        GameObject leftArm = createTwoBoneIKConstraint("leftArm", rig1.transform);
        GameObject rightArm = createTwoBoneIKConstraint("rightArm", rig1.transform);
        GameObject ls = createChainIKConstraint("leftShoulder", rig1.transform);
        GameObject rs = createChainIKConstraint("rightShoulder", rig1.transform);
        GameObject spine = createChainIKConstraint("spine", rig1.transform);

        TwoBoneIKConstraint leftArmConstraint = leftArm.GetComponent<TwoBoneIKConstraint>();
        leftArmConstraint.data.root = leftShoulder;
        leftArmConstraint.data.mid = leftElbow;
        leftArmConstraint.data.tip = leftWrist;

        leftArmConstraint.data.target = targetThreeDPoints[13].transform;
        leftArmConstraint.data.hint = targetThreeDPoints[12].transform;

        leftArmConstraint.data.targetPositionWeight = 1.0f;
        leftArmConstraint.data.targetRotationWeight = 1.0f;
        leftArmConstraint.data.hintWeight = 1.0f;

        TwoBoneIKConstraint rightArmConstraint = rightArm.GetComponent<TwoBoneIKConstraint>();
        rightArmConstraint.data.root = rightShoulder;
        rightArmConstraint.data.mid = rightElbow;
        rightArmConstraint.data.tip = rightWrist;

        rightArmConstraint.data.target = targetThreeDPoints[16].transform;
        rightArmConstraint.data.hint = targetThreeDPoints[15].transform;

        rightArmConstraint.data.targetPositionWeight = 1.0f;
        rightArmConstraint.data.targetRotationWeight = 1.0f;
        rightArmConstraint.data.hintWeight = 1.0f;

        MultiAimConstraint lookAtConstraint = lookAt.GetComponent<MultiAimConstraint>();
        lookAtConstraint.data.constrainedObject = nose;
        var sources = lookAtConstraint.data.sourceObjects;
        sources.Add(new WeightedTransform(targetThreeDPoints[9].transform, 0.3f));

        lookAtConstraint.data.sourceObjects = sources;
        lookAtConstraint.data.aimAxis = MultiAimConstraintData.Axis.Y_NEG;
        lookAtConstraint.data.upAxis = MultiAimConstraintData.Axis.Y;
        lookAtConstraint.data.maintainOffset = true;
        lookAtConstraint.data.constrainedXAxis = true;
        lookAtConstraint.data.constrainedYAxis = true;
        lookAtConstraint.data.constrainedZAxis = true;
        lookAtConstraint.data.limits = new Vector2(-60,60);

        ChainIKConstraint lsConstraint = ls.GetComponent<ChainIKConstraint>();
        lsConstraint.data.root = belly;
        lsConstraint.data.tip = leftShoulder;
        lsConstraint.data.target = targetThreeDPoints[11].transform;
        lsConstraint.data.maxIterations = 10;
        lsConstraint.data.tolerance = 0.001f;
        lsConstraint.data.chainRotationWeight = 0.5f;

        ChainIKConstraint rsConstraint = rs.GetComponent<ChainIKConstraint>();
        rsConstraint.data.root = belly;
        rsConstraint.data.tip = rightShoulder;
        rsConstraint.data.target = targetThreeDPoints[14].transform;
        rsConstraint.data.maxIterations = 10;
        rsConstraint.data.tolerance = 0.001f;
        rsConstraint.data.chainRotationWeight = 0.5f;

        rigBuilder.Build();

        boneDistances = saveBoneDistances();

    }

    private float[] saveBoneDistances() {

        float[] bones = new float[16];

        bones[0] = distAtoB(characterRoot.transform.position, belly.transform.position);
        bones[1] = distAtoB(belly.transform.position, neck.transform.position);
        bones[2] = distAtoB(neck.transform.position, nose.transform.position);
        bones[3] = distAtoB(neck.transform.position, rightShoulder.transform.position);
        bones[4] = distAtoB(rightShoulder.transform.position, rightElbow.transform.position);
        bones[5] = distAtoB(rightElbow.transform.position, rightWrist.transform.position);
        bones[6] = distAtoB(neck.transform.position, leftShoulder.transform.position);
        bones[7] = distAtoB(leftShoulder.transform.position, leftElbow.transform.position);
        bones[8] = distAtoB(leftElbow.transform.position, leftWrist.transform.position);

        return bones;

    }

    private float distAtoB(Vector3 a, Vector3 b) {

        float dist = Vector3.Distance(a,b);

        return dist;

    }

    private Vector3 fromAtoB(Vector3 a, Vector3 b) {

        Vector3 ab = b - a;

        return ab;

    }

    private GameObject createTwoBoneIKConstraint(string name, Transform rig) {

        GameObject constraintObject = new GameObject(name);

        constraintObject.AddComponent<TwoBoneIKConstraint>();

        constraintObject.transform.SetParent(rig);

        return constraintObject;

    }

    private GameObject createChainIKConstraint(string name, Transform rig) {

        GameObject constraintObject = new GameObject(name);

        constraintObject.AddComponent<ChainIKConstraint>();

        constraintObject.transform.SetParent(rig);

        return constraintObject;

    }

    private GameObject createMultiAimConstraint(string name, Transform rig) {

        GameObject constraintObject = new GameObject(name);

        constraintObject.AddComponent<MultiAimConstraint>();

        constraintObject.transform.SetParent(rig);

        return constraintObject;

    }

    void OnDestroy() {

        poseEstimator.Dispose();

    }

}
