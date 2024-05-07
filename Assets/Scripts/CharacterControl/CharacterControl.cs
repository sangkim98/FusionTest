using UnityEngine;
using System.Collections.Generic;
using System;
using Unity.Sentis;

public class CharacterControl : MonoBehaviour {
    // Pose Estimator
    WebCamTexture webcamTexture;
    PoseEstimator poseEstimator;
    const int resizedSquareImageDim = 320;
    // Game Objects
    private List<GameObject> points3d;
    private Animator animator;
    private float hipHeadEndDistance;
    
    public ModelAsset twoDPoseModelAsset;
    public ModelAsset threeDPoseModelAsset;

    // IK Control
    public bool ikActive = false;
    private Transform leftHandObj = null;
    private Transform rightHandObj = null;
    private Transform lookObj = null;

    void Start() {

        WebCamDevice[] devices = WebCamTexture.devices;
        
        webcamTexture = new WebCamTexture(devices[0].name, 640, 360, 60);

        webcamTexture.Play();

        // Sentis initialization
        poseEstimator = new PoseEstimator(resizedSquareImageDim, ref twoDPoseModelAsset, ref threeDPoseModelAsset, BackendType.GPUCompute);

        // Animations

        init3DKeypoints();

        animator = GetComponent<Animator>();

        Transform headEndTransform = animator.transform.Find(
            "mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/" +
            "mixamorig:Spine2/mixamorig:Neck/mixamorig:Head/mixamorig:HeadTop_End"
        );
        Transform hipTransform = animator.transform.Find(
            "mixamorig:Hips"
        );

        hipHeadEndDistance = Vector3.Distance(hipTransform.position, headEndTransform.position);

        lookObj = GameObject.Find("point3d_9").transform;
        rightHandObj = GameObject.Find("point3d_16").transform;
        leftHandObj = GameObject.Find("point3d_13").transform;

    }

    void Update() {

        bool goodEstimate;

        goodEstimate = poseEstimator.RunML(webcamTexture);

        if(goodEstimate) {

            Vector3[] threeDJoints = poseEstimator.getThreeDPose();

            scaleTranslateJoints(threeDJoints);

            Draw3DPoints(threeDJoints);

        }

    }

    void OnAnimatorIK() {

        if(animator) {

            if(ikActive) {

                if(lookObj != null) {
                    animator.SetLookAtWeight(1);
                    animator.SetLookAtPosition(lookObj.position);
                }    

                if(leftHandObj != null) {
                    animator.SetIKPositionWeight(AvatarIKGoal.LeftHand,1);
                    animator.SetIKRotationWeight(AvatarIKGoal.LeftHand,1);
                    animator.SetIKPosition(AvatarIKGoal.LeftHand, leftHandObj.position);
                    animator.SetIKRotation(AvatarIKGoal.LeftHand, leftHandObj.rotation);
                }

                if(rightHandObj != null) {
                    animator.SetIKPositionWeight(AvatarIKGoal.RightHand,1);
                    animator.SetIKRotationWeight(AvatarIKGoal.RightHand,1);
                    animator.SetIKPosition(AvatarIKGoal.RightHand, rightHandObj.position);
                    animator.SetIKRotation(AvatarIKGoal.RightHand, rightHandObj.rotation);
                }

            }

            else {

                animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, 0);
                animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, 0);
                animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 0);
                animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 0);
                animator.SetLookAtWeight(0);

            }

        }

    }

    private void init3DKeypoints() {

        points3d = new List<GameObject>();

        for (int i = 0; i < 17; i++) {

            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            sphere.name = String.Format("point3d_{0}", i);

            sphere.transform.localScale = new Vector3(0.1f,0.1f,0.1f);
            sphere.transform.localPosition = new Vector3(0,0,0);

            sphere.transform.SetParent(transform, false);

            points3d.Add(sphere);

        }

    }

    private void scaleTranslateJoints(Vector3[] joints) {

        const int rootIndex = 0;
        const int headIndex = 10;

        Transform hipTransform = animator.transform.Find(
            "mixamorig:Hips"
        );

        Vector3 delta = getRootHipDelta(hipTransform.position, joints[rootIndex]);
        float rootHeadDistance = getRootHeadDistance(joints[rootIndex], joints[headIndex]);
        float ratio = 1.2f * (hipHeadEndDistance / rootHeadDistance);

        for(int idx = 0; idx < joints.Length; idx++) {

            joints[idx] *= ratio;
            joints[idx] += delta;

        }

    }

    private float getRootHeadDistance(Vector3 root, Vector3 head) {

        return Vector3.Distance(root, head);

    }

    private Vector3 getRootHipDelta(Vector3 root, Vector3 hip) {

        return root - hip;

    }

    void Draw3DPoints(Vector3[] joints) {

        for (int idx = 0; idx < joints.Length; idx++) {

            GameObject point = points3d[idx];
            point.transform.localPosition = joints[idx];

        }

    }

    void OnDestroy() {

        poseEstimator.Dispose();

    }

}
