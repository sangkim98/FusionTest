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
