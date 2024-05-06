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
    private GameObject targetCharacter;
    private List<GameObject> points3d;
    private Animator animator;
    private float hipHeadEndDistance;
    private Vector3 hipLocation;
    
    public ModelAsset twoDPoseModelAsset;
    public ModelAsset threeDPoseModelAsset;

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
            "Armature/mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/" +
            "mixamorig:Spine2/mixamorig:Neck/mixamorig:Head/mixamorig:Head_end"
        );
        Transform hipTransform = animator.transform.Find(
            "Armature/mixamorig:Hips"
        );

        hipLocation = hipTransform.position;

        hipHeadEndDistance = Vector3.Distance(hipTransform.position, headEndTransform.position);

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

    private void init3DKeypoints() {

        points3d = new List<GameObject>();

        targetCharacter = GameObject.Find("CharacterAnimated");

        for (int i = 0; i < 17; i++) {

            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            sphere.name = String.Format("point3d_{0}", i);

            sphere.transform.localScale = new Vector3(1,1,1);
            sphere.transform.localPosition = new Vector3(0,0,0);

            sphere.transform.SetParent(targetCharacter.transform, false);

            points3d.Add(sphere);

        }

    }

    private void scaleTranslateJoints(Vector3[] joints) {

        const int rootIndex = 0;
        const int headIndex = 10;

        Vector3 delta = getRootHipDelta(hipLocation, joints[rootIndex]);
        float rootHeadDistance = getRootHeadDistance(joints[rootIndex], joints[headIndex]);
        float ratio = hipHeadEndDistance / rootHeadDistance;

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
