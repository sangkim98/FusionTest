using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;
using System.Collections.Generic;
using System;

public class PoseEstimatorTest : MonoBehaviour
{
    // Game Objects
    private List<GameObject> points;
    private List<GameObject> points3d;
    private GameObject canvasObject;
    private GameObject poseImage;
    private GameObject points3dGroup;
    private Animator animator;
    private const int scale3d = 150;
    // Sprites and Materials
    public Sprite spriteCircle;
    public Material webcamMaterial;

    private PoseEstimator poseEstimator;

    WebCamTexture webcamTexture;

    // Webcam Setup
    const int webcamWidth = 640;
    const int webcamHeight = 360;

    const int resizedSquareImageDim = 320;
    
    // Sentis Related
    public ModelAsset yoloPoseModelAsset;
    public ModelAsset motionbertModelAsset;
    TensorFloat twoDJointsTensor;
    TensorFloat threeDJointsTensor;

    // [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    // [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    void Start()
    {

        WebCamDevice[] devices = WebCamTexture.devices;

        webcamTexture = new WebCamTexture(devices[0].name, webcamWidth, webcamHeight, 60);

        webcamTexture.Play();

        poseEstimator = new PoseEstimator(resizedSquareImageDim, ref yoloPoseModelAsset, ref motionbertModelAsset, backend: BackendType.GPUCompute);

        points = new List<GameObject>();
        points3d = new List<GameObject>();
        
        initSprites();
        init3DKeypoints();
        initPoseImage();

        animator = GetComponent<Animator>();

        Transform shoulderTransform = animator.transform.Find("Armature/mixamorig:Hips");

        if (shoulderTransform != null) {
            Debug.Log("Shoulder position: " + shoulderTransform.position);
        }
    }

    void Update()
    {

        poseEstimator.executeTwoDPoseWorker(webcamTexture);

        twoDJointsTensor = poseEstimator.getTwoDPoseOutput();

        if (twoDJointsTensor.shape[2] == 17 && twoDJointsTensor.shape[3] == 3) {

            float[] TwoDJointsForDisplay = twoDJointsTensor.ToReadOnlyArray();

            Draw2DPoints(TwoDJointsForDisplay);

            poseEstimator.executeThreeDPoseWorker(twoDJointsTensor);

            threeDJointsTensor = poseEstimator.getThreeDPoseOutput();
            
            float[] ThreeDJointsForDisplay = threeDJointsTensor.ToReadOnlyArray();

            Draw3DPoints(ThreeDJointsForDisplay);

            threeDJointsTensor?.Dispose();

        }

        twoDJointsTensor?.Dispose();

    }

    void initPoseImage() {

        GameObject canvas = GameObject.Find("Canvas");
        poseImage = new GameObject("Webcam Image") { layer = 5 };

        poseImage.AddComponent<CanvasRenderer>();
        RectTransform rectTransformImage = poseImage.AddComponent<RectTransform>();
        Image image = poseImage.AddComponent<Image>();

        poseImage.transform.SetParent(canvas.transform, false);
        poseImage.transform.localPosition = new Vector3(0,0,0);

        rectTransformImage.sizeDelta = new Vector2(640, 360);

        image.material = webcamMaterial;
        image.material.mainTexture = webcamTexture;
        image.material.shader = Shader.Find("UI/Unlit/Detail");

    }

    void initSprites() {

        canvasObject = new GameObject("Canvas") { layer = 5 };

        RectTransform rectTransformCanvas = canvasObject.AddComponent<RectTransform>();
        rectTransformCanvas.position = new Vector3(0, 200, 0);
        rectTransformCanvas.sizeDelta = new Vector2(640, 360);

        Canvas canvas = canvasObject.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        CanvasScaler canvasScaler = canvasObject.AddComponent<CanvasScaler>();
        canvasScaler.uiScaleMode = CanvasScaler.ScaleMode.ConstantPhysicalSize;

        canvasObject.AddComponent<GraphicRaycaster>();

        for (int i = 0; i < 17; i++) {

            GameObject spriteObject = new GameObject(String.Format("point_{0}", i)){ layer = 6 };
            points.Add(spriteObject);
            spriteObject.transform.SetParent(canvas.transform, false);
            SpriteRenderer spriteRenderer = spriteObject.AddComponent<SpriteRenderer>();
            spriteRenderer.sprite = spriteCircle;
            spriteRenderer.sortingOrder = 1;
            spriteObject.transform.localPosition = new Vector3(0,0,0);
            spriteObject.transform.localScale = new Vector3(10,10,10);
            spriteRenderer.color = new Color(
                UnityEngine.Random.Range(0.1f, 0.8f),
                UnityEngine.Random.Range(0.1f, 0.6f),
                UnityEngine.Random.Range(0.1f, 0.7f)
            );

        }
    }

    void init3DKeypoints() {

        points3dGroup = new GameObject("Points 3D Group"){ layer = 6 };

        points3dGroup.transform.position = new Vector3(0, 150, -200);

        for (int i = 0; i < 17; i++) {

            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);

            sphere.name = String.Format("point3d_{0}", i);

            sphere.transform.localScale = new Vector3(5,5,5);
            sphere.transform.localPosition = new Vector3(0,0,0);

            sphere.transform.SetParent(points3dGroup.transform, false);

            points3d.Add(sphere);

        }

    }

    void Draw3DPoints(float[] joints) {

        int arrlen = joints.Length;

        for (int idx = 0; idx < arrlen; idx += 3) {
            float x = joints[idx] * scale3d;
            float y = joints[idx+1] * scale3d;
            float z = joints[idx+2] * scale3d;

            GameObject point = points3d[idx/3];
            point.transform.localPosition = new Vector3(x, -y, z);
        }

    }
    
    void Draw2DPoints(float[] joints) {

        int arrlen = joints.Length;

        for (int idx = 0; idx < arrlen; idx += 3) {

            float x = joints[idx];
            float y = joints[idx+1];

            if (x >=0 && x < resizedSquareImageDim && y >= 0 && y < resizedSquareImageDim) {
                DrawPoint(idx, x, y);
            }

        }

    }

    void DrawPoint(int index, float x, float y) {

        GameObject point = points[index / 3];
        x = (x-(resizedSquareImageDim / 2)) * ((float)webcamTexture.width / resizedSquareImageDim);
        y = (y-(resizedSquareImageDim / 2)) * ((float)webcamTexture.height / resizedSquareImageDim);
        point.transform.localPosition = new Vector3(x, -y, 0);

    }

    void OnDistroy() {

        twoDJointsTensor?.Dispose();
        threeDJointsTensor?.Dispose();
        poseEstimator?.Dispose();

        twoDJointsTensor = null;
        threeDJointsTensor = null;
        poseEstimator = null;

    }

}
