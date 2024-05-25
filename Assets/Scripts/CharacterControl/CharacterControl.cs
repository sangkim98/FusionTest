using UnityEngine;
using System.Collections.Generic;
using System;
using Unity.Sentis;
using UnityEngine.Animations.Rigging;
using UnityEngine.Animations;
using System.Drawing;

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
    private Rig core;
    private Rig arms;
    private Rig look;
    // For joint control
    public Transform characterRoot;
    public Transform belly;
    public Transform spine;
    public Transform head;
    public Transform neck;
    public Transform nose;
    public Transform leftShoulder;
    public Transform leftArm;
    public Transform leftForeArm;
    public Transform leftWrist;
    public Transform rightShoulder;
    public Transform rightArm;
    public Transform rightForeArm;
    public Transform rightWrist;

    private float[] boneDistances;

    void Start() {

        WebCamDevice[] devices = WebCamTexture.devices;
        
        webcamTexture = new WebCamTexture(devices[0].name, 640, 360, 60);

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
        GameObject root = new GameObject("pose_root");
        root.transform.SetParent(characterRoot);
        root.transform.localPosition = Vector3.zero;
        root.transform.Rotate(0,0,0);

        for (int i = 0; i <= 18; i++) {

            GameObject sphere = new GameObject(String.Format("joint{0}", i));

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

        joints[0].x = 0;
        joints[0].y = 0;
        joints[0].z = 0;
        joints[7] = joints[0] + rootToBelly * boneDistances[0];
        joints[8] = joints[7] + bellyToNeck * boneDistances[1];
        joints[9] = joints[8] + neckToNose * boneDistances[2];
        joints[10] = joints[9] + noseToHead * 1.5f * boneDistances[2];
        joints[9] = joints[8] + neckToNose * boneDistances[2];
        joints[11] = joints[8] + neckToLeftShoulder * boneDistances[6];
        joints[12] = joints[11] + leftShoulderToElbow * boneDistances[7];
        joints[13] = joints[12] + leftElbowToWrist * boneDistances[8];
        var leftHandPoint = joints[12] + 1.2f * leftElbowToWrist * boneDistances[8];
        joints[14] = joints[8] + neckToRightShoulder * boneDistances[3];
        joints[15] = joints[14] + rightShoulderToElbow * boneDistances[4];
        joints[16] = joints[15] + rightElbowToWrist * boneDistances[5];
        var rightHandPoint = joints[15] + 1.2f * rightElbowToWrist * boneDistances[5];
        joints[8] = joints[7] + 0.8f * bellyToNeck * boneDistances[1];
        joints[9] = joints[8] + neckToNose * boneDistances[2];

        for (int idx = 0; idx < joints.Length; idx++) {

            GameObject point = targetThreeDPoints[idx];
            point.transform.localPosition = joints[idx];

        }

        targetThreeDPoints[17].transform.localPosition = leftHandPoint;
        targetThreeDPoints[18].transform.localPosition = rightHandPoint;

    }

    void SetupRig() {

        RigBuilder rigBuilder = gameObject.AddComponent<RigBuilder>();
        GameObject rig_core = new GameObject("Core Rig");
        GameObject rig_arms = new GameObject("Arm Rig");
        GameObject rig_look = new GameObject("Look Rig");
        core = rig_core.AddComponent<Rig>();
        arms = rig_arms.AddComponent<Rig>();
        look = rig_look.AddComponent<Rig>();
        core.weight = weight;
        arms.weight = weight;
        look.weight = weight;

        rig_core.transform.SetParent(gameObject.transform);
        rig_arms.transform.SetParent(gameObject.transform);
        rig_look.transform.SetParent(gameObject.transform);

        rigBuilder.layers.Add(new RigLayer(core.GetComponent<Rig>(), true));
        rigBuilder.layers.Add(new RigLayer(arms.GetComponent<Rig>(), true));
        rigBuilder.layers.Add(new RigLayer(look.GetComponent<Rig>(), true));

        // Setup Core Rig
        GameObject spineControl = new GameObject("spineControl");
        GameObject lookAt = new GameObject("lookAt");
        GameObject neckControl = new GameObject("neckControl");

        spineControl.transform.SetParent(rig_core.transform);
        neckControl.transform.SetParent(spineControl.transform);

        // Main Spine Constraints
        var spine_chainIK = spineControl.AddComponent<ChainIKConstraint>();
        var spine_multiAim = spineControl.AddComponent<MultiAimConstraint>();
        var spine_multiRotation = spineControl.AddComponent<MultiRotationConstraint>();

        // Chain IK
        spine_chainIK.data.root = belly;
        spine_chainIK.data.tip = spine;
        spine_chainIK.data.target = targetThreeDPoints[8].transform;
        spine_chainIK.data.maxIterations = 15;
        spine_chainIK.data.tolerance = 0.0001f;
        spine_chainIK.data.chainRotationWeight = 1.0f;
        spine_chainIK.data.tipRotationWeight = 0.0f;
        // Multi-Aim
        spine_multiAim.data.constrainedObject = spine;

        var sources1 = spine_multiAim.data.sourceObjects;
        sources1.Add(new WeightedTransform(targetThreeDPoints[11].transform, 1.0f));
        sources1.Add(new WeightedTransform(targetThreeDPoints[14].transform, 1.0f));
        spine_multiAim.data.sourceObjects = sources1;

        spine_multiAim.data.aimAxis = MultiAimConstraintData.Axis.Z;
        spine_multiAim.data.upAxis = MultiAimConstraintData.Axis.Y;
        spine_multiAim.data.maintainOffset = false;
        spine_multiAim.data.constrainedXAxis = true;
        spine_multiAim.data.constrainedYAxis = true;
        spine_multiAim.data.constrainedZAxis = true;
        spine_multiAim.data.limits = new Vector2(-60,60);
        // Multi-Rotation
        
        // Look Rig Setup
        // Look At Contraints
        lookAt.transform.SetParent(rig_look.transform);

        var lookAt_multiAim = lookAt.AddComponent<MultiAimConstraint>();
        lookAt_multiAim.weight = 1.0f;
        lookAt_multiAim.data.constrainedObject = nose;
        lookAt_multiAim.data.aimAxis = MultiAimConstraintData.Axis.Z;
        lookAt_multiAim.data.upAxis = MultiAimConstraintData.Axis.Y;

        var sources2 = lookAt_multiAim.data.sourceObjects;
        sources2.Add(new WeightedTransform(targetThreeDPoints[9].transform, 1.0f));
        lookAt_multiAim.data.sourceObjects = sources2;

        lookAt_multiAim.data.maintainOffset = false;
        lookAt_multiAim.data.constrainedXAxis = true;
        lookAt_multiAim.data.constrainedYAxis = true;
        lookAt_multiAim.data.constrainedZAxis = true;
        lookAt_multiAim.data.limits = new Vector2(-60,60);

        // Neck Contraints
        var neck_twoBone = neckControl.AddComponent<TwoBoneIKConstraint>();

        neck_twoBone.data.root = spine;
        neck_twoBone.data.mid = neck;
        neck_twoBone.data.tip = head;

        neck_twoBone.data.target = targetThreeDPoints[10].transform;
        neck_twoBone.data.hint = targetThreeDPoints[9].transform;

        neck_twoBone.data.targetPositionWeight = 1.0f;
        neck_twoBone.data.targetRotationWeight = 0.0f;
        neck_twoBone.data.hintWeight = 0.0f;

        // Setup Arm Rig
        GameObject leftShoulderControl = new GameObject("leftShoulderControl");
        GameObject leftArmControl = new GameObject("leftArmControl");
        GameObject leftHandControl = new GameObject("leftHandControl");
        GameObject rightShoulderControl = new GameObject("rightShoulderControl");
        GameObject rightArmControl = new GameObject("rightArmControl");
        GameObject rightHandControl = new GameObject("rightHandControl");

        leftShoulderControl.transform.SetParent(rig_arms.transform);
        leftArmControl.transform.SetParent(leftShoulderControl.transform);
        leftHandControl.transform.SetParent(leftArmControl.transform);

        rightShoulderControl.transform.SetParent(rig_arms.transform);
        rightArmControl.transform.SetParent(rightShoulderControl.transform);
        rightHandControl.transform.SetParent(rightArmControl.transform);

        // Setup Left
        // Left Shoulder
        var leftShoulder_twoBone = leftShoulderControl.AddComponent<TwoBoneIKConstraint>();

        leftShoulder_twoBone.data.root = leftShoulder;
        leftShoulder_twoBone.data.mid = leftArm;
        leftShoulder_twoBone.data.tip = leftForeArm;

        leftShoulder_twoBone.data.target = targetThreeDPoints[12].transform;
        leftShoulder_twoBone.data.hint = targetThreeDPoints[11].transform;

        leftShoulder_twoBone.data.targetPositionWeight = 0.2f;
        leftShoulder_twoBone.data.targetRotationWeight = 0.0f;
        leftShoulder_twoBone.data.hintWeight = 0.7f;

        // Left Arm
        var leftArm_twoBone = leftArmControl.AddComponent<TwoBoneIKConstraint>();

        leftArm_twoBone.data.root = leftArm;
        leftArm_twoBone.data.mid = leftForeArm;
        leftArm_twoBone.data.tip = leftWrist;

        leftArm_twoBone.data.target = targetThreeDPoints[13].transform;
        leftArm_twoBone.data.hint = targetThreeDPoints[12].transform;

        leftArm_twoBone.data.targetPositionWeight = 1.0f;
        leftArm_twoBone.data.targetRotationWeight = 0.0f;
        leftArm_twoBone.data.hintWeight = 1.0f;

        // Left Hand
        var leftHand_multiAim = leftHandControl.AddComponent<MultiAimConstraint>();

        leftHand_multiAim.weight = 1.0f;
        leftHand_multiAim.data.constrainedObject = leftWrist;
        leftHand_multiAim.data.aimAxis = MultiAimConstraintData.Axis.Y;
        leftHand_multiAim.data.upAxis = MultiAimConstraintData.Axis.Y;

        var sources3 = leftHand_multiAim.data.sourceObjects;
        sources3.Add(new WeightedTransform(targetThreeDPoints[17].transform, 1.0f));
        leftHand_multiAim.data.sourceObjects = sources3;

        leftHand_multiAim.data.maintainOffset = false;
        leftHand_multiAim.data.constrainedXAxis = true;
        leftHand_multiAim.data.constrainedYAxis = true;
        leftHand_multiAim.data.constrainedZAxis = true;
        leftHand_multiAim.data.limits = new Vector2(0,0);

        // Right Shoulder
        var rightShoulder_twoBone = rightShoulderControl.AddComponent<TwoBoneIKConstraint>();

        rightShoulder_twoBone.data.root = rightShoulder;
        rightShoulder_twoBone.data.mid = rightArm;
        rightShoulder_twoBone.data.tip = rightForeArm;

        rightShoulder_twoBone.data.target = targetThreeDPoints[15].transform;
        rightShoulder_twoBone.data.hint = targetThreeDPoints[14].transform;

        rightShoulder_twoBone.data.targetPositionWeight = 0.2f;
        rightShoulder_twoBone.data.targetRotationWeight = 0.0f;
        rightShoulder_twoBone.data.hintWeight = 0.7f;

        // Right Arm
        var rightArm_twoBone = rightArmControl.AddComponent<TwoBoneIKConstraint>();

        rightArm_twoBone.data.root = rightArm;
        rightArm_twoBone.data.mid = rightForeArm;
        rightArm_twoBone.data.tip = rightWrist;

        rightArm_twoBone.data.target = targetThreeDPoints[16].transform;
        rightArm_twoBone.data.hint = targetThreeDPoints[15].transform;

        rightArm_twoBone.data.targetPositionWeight = 1.0f;
        rightArm_twoBone.data.targetRotationWeight = 0.0f;
        rightArm_twoBone.data.hintWeight = 1.0f;

        // Right Hand
        var rightHand_multiAim = rightHandControl.AddComponent<MultiAimConstraint>();

        rightHand_multiAim.weight = 1.0f;
        rightHand_multiAim.data.constrainedObject = rightWrist;
        rightHand_multiAim.data.aimAxis = MultiAimConstraintData.Axis.Y;
        rightHand_multiAim.data.upAxis = MultiAimConstraintData.Axis.Y;

        var sources4 = rightHand_multiAim.data.sourceObjects;
        sources4.Add(new WeightedTransform(targetThreeDPoints[18].transform, 1.0f));
        rightHand_multiAim.data.sourceObjects = sources3;

        rightHand_multiAim.data.maintainOffset = false;
        rightHand_multiAim.data.constrainedXAxis = true;
        rightHand_multiAim.data.constrainedYAxis = true;
        rightHand_multiAim.data.constrainedZAxis = true;
        rightHand_multiAim.data.limits = new Vector2(0,0);

        // Build Rig
        rigBuilder.Build();

        boneDistances = saveBoneDistances();

    }

    private float[] saveBoneDistances() {

        float[] bones = new float[16];

        bones[0] = distAtoB(characterRoot.transform.position, belly.transform.position);
        bones[1] = distAtoB(belly.transform.position, neck.transform.position);
        bones[2] = distAtoB(neck.transform.position, nose.transform.position);
        bones[3] = distAtoB(neck.transform.position, rightArm.transform.position);
        bones[4] = distAtoB(rightArm.transform.position, rightForeArm.transform.position);
        bones[5] = distAtoB(rightForeArm.transform.position, rightWrist.transform.position);
        bones[6] = distAtoB(neck.transform.position, leftArm.transform.position);
        bones[7] = distAtoB(leftArm.transform.position, leftForeArm.transform.position);
        bones[8] = distAtoB(leftForeArm.transform.position, leftWrist.transform.position);
        bones[9] = distAtoB(nose.transform.position, head.transform.position);

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

    void OnDestroy() {

        poseEstimator.Dispose();

    }

}
