using UnityEngine;
using System.Collections.Generic;
using System;
using Unity.Sentis;
using UnityEngine.Animations.Rigging;

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

    // For Model Scaling
    public Transform characterRoot;

    // IK Control
    public bool followPose = false;
    public bool lowerBody = false;

    public Transform root;
    public Transform rightHip;
    public Transform rightKnee;
    public Transform rightAnkle;
    public Transform leftHip;
    public Transform leftKnee;
    public Transform leftAnkle;
    public Transform belly;
    public Transform neck;
    public Transform nose;
    public Transform head;
    public Transform leftShoulder;
    public Transform leftElbow;
    public Transform leftWrist;
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

    }
    private void init3DKeypoints() {

        targetThreeDPoints = new List<GameObject>();

        for (int i = 0; i < 17; i++) {

            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            sphere.name = String.Format("joint{0}", i);

            sphere.transform.localScale = new Vector3(0.05f,0.05f,0.05f);
            sphere.transform.localPosition = new Vector3(0,0,0);

            sphere.transform.SetParent(transform, false);

            targetThreeDPoints.Add(sphere);

        }

    }

    void Draw3DPoints(Vector3[] joints) {

        Vector3 rootToBelly = fromAtoB(joints[0], joints[7]);
        Vector3 bellyToNeck = fromAtoB(joints[7], joints[8]);
        Vector3 neckToNose = fromAtoB(joints[8], joints[9]);
        Vector3 noseToHead = fromAtoB(joints[9], joints[10]);
        Vector3 neckToLeftShoulder = fromAtoB(joints[8], joints[11]);
        Vector3 leftShoulderToElbow = fromAtoB(joints[11], joints[12]);
        Vector3 leftElbowToWrist = fromAtoB(joints[12], joints[13]);
        Vector3 neckToRightShoulder = fromAtoB(joints[8], joints[14]);
        Vector3 rightShoulderToElbow = fromAtoB(joints[14], joints[15]);
        Vector3 rightElbowToWrist = fromAtoB(joints[15], joints[16]);

        rootToBelly.Normalize();
        bellyToNeck.Normalize();
        neckToNose.Normalize();
        noseToHead.Normalize();
        neckToLeftShoulder.Normalize();
        leftShoulderToElbow.Normalize();
        leftElbowToWrist.Normalize();
        neckToRightShoulder.Normalize();
        rightShoulderToElbow.Normalize();
        rightElbowToWrist.Normalize();

        joints[7] = joints[0] + rootToBelly * boneDistances[2];
        joints[8] = joints[7] + bellyToNeck * boneDistances[3];
        joints[9] = joints[8] + neckToNose * boneDistances[4];
        joints[10] = joints[9] + noseToHead * boneDistances[5];
        joints[11] = joints[8] + neckToLeftShoulder * boneDistances[9];
        joints[12] = joints[11] + leftShoulderToElbow * boneDistances[10];
        joints[13] = joints[12] + leftElbowToWrist * boneDistances[11];
        joints[14] = joints[8] + neckToRightShoulder * boneDistances[6];
        joints[15] = joints[14] + rightShoulderToElbow * boneDistances[7];
        joints[16] = joints[15] + rightElbowToWrist * boneDistances[8];

        for (int idx = 0; idx < joints.Length; idx++) {

            GameObject point = targetThreeDPoints[idx];
            point.transform.localPosition = joints[idx];

        }

    }

    void SetupRig() {

        RigBuilder rigBuilder = gameObject.AddComponent<RigBuilder>();
        GameObject rig1 = new GameObject("Rig1");
        rig1.AddComponent<Rig>();

        rig1.transform.SetParent(gameObject.transform);

        rigBuilder.layers.Add(new RigLayer(rig1.GetComponent<Rig>(), true));

        createMultiAimConstraint("lookAt", rig1.transform);
        createMultiAimConstraint("shoulder", rig1.transform);
        createTwoBoneIKConstraint("leftArm", rig1.transform);
        createTwoBoneIKConstraint("rightArm", rig1.transform);
        createChainIKConstraint("spine", rig1.transform);

        boneDistances = saveBoneDistances();

    }

    private float[] saveBoneDistances() {

        float[] bones = new float[16];

        bones[0] = distAtoB(root.transform.localPosition, rightHip.transform.localPosition);
        bones[1] = distAtoB(root.transform.localPosition, leftHip.transform.localPosition);
        bones[2] = distAtoB(root.transform.localPosition, belly.transform.localPosition);
        bones[3] = distAtoB(belly.transform.localPosition, neck.transform.localPosition);
        bones[4] = distAtoB(neck.transform.localPosition, nose.transform.localPosition);
        bones[5] = distAtoB(nose.transform.localPosition, head.transform.localPosition);
        bones[6] = distAtoB(neck.transform.localPosition, rightShoulder.transform.localPosition);
        bones[7] = distAtoB(rightShoulder.transform.localPosition, rightElbow.transform.localPosition);
        bones[8] = distAtoB(rightElbow.transform.localPosition, rightWrist.transform.localPosition);
        bones[9] = distAtoB(neck.transform.localPosition, leftShoulder.transform.localPosition);
        bones[10] = distAtoB(leftShoulder.transform.localPosition, leftElbow.transform.localPosition);
        bones[11] = distAtoB(leftElbow.transform.localPosition, leftWrist.transform.localPosition);

        for(int i = 0; i < 12; i++) {
            Debug.Log(bones[i]);
        }

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
