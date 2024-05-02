using UnityEngine;
using Unity.Sentis;
using FF = Unity.Sentis.Functional;
using Unity.VisualScripting;
using UnityEngine.UI;
using System.Collections.Generic;
using System;

public class PoseEstimatorTest : MonoBehaviour
{
    // Game Objects
    private List<GameObject> points;
    private List<GameObject> points3d;
    private GameObject poseImage;
    private GameObject points3dGroup;
    private const int scale3d = 150;
    // Sprites and Materials
    public Sprite spriteCircle;
    public Material webcamMaterial;

    private PoseEstimator poseEstimator;

    WebCamTexture webcamTexture;

    // Webcam Setup
    const int webcamWidth = 640;
    const int webcamHeight = 360;

    const int resizedImageSize = 320;
    private float scaleX;
    private float scaleY;
    // Sentis Related
    public ModelAsset yoloPoseModelAsset;
    public ModelAsset motionbertModelAsset;
    
    // [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    // [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        for (int i = 0; i < devices.Length; i++)
        {
            print("Webcam available: " + devices[i].name);
        }

        webcamTexture = new WebCamTexture(devices[1].name, webcamWidth, webcamHeight, 60);

        webcamTexture.Play();

        poseEstimator = new PoseEstimator(webcamTexture, ref yoloPoseModelAsset, ref motionbertModelAsset, backend: BackendType.GPUCompute);

        (scaleX, scaleY) = poseEstimator.getImageScale();

        points = new List<GameObject>();
        points3d = new List<GameObject>();

        initSprites();
        init3DKeypoints();
        initPoseImage();
    }

    void Update()
    {        
        poseEstimator.executeTwoDPoseWorker(webcamTexture);

        TensorFloat joints_yolo = poseEstimator.getTwoDPoseOutput();

        DrawPoints(joints_yolo);

        poseEstimator.executeThreeDPoseWorker(joints_yolo);

        TensorFloat joints_3d = poseEstimator.getThreeDPoseOutput();

        Debug.Log("3D Pose Shape: " + joints_3d.shape);

        Draw3dPoints(joints_3d);

        joints_yolo.Dispose();
        joints_3d.Dispose();
    }

    void initPoseImage() {
        GameObject canvas = GameObject.Find("Canvas");
        poseImage = new GameObject("Webcam Image") { layer = 5 };

        poseImage.AddComponent<CanvasRenderer>();
        RectTransform rectTransform = poseImage.AddComponent<RectTransform>();
        Image image = poseImage.AddComponent<Image>();

        poseImage.transform.SetParent(canvas.transform, false);
        poseImage.transform.localPosition = new Vector3(0,0,0);

        rectTransform.sizeDelta = new Vector2(640, 360);

        image.material = webcamMaterial;
        image.material.mainTexture = webcamTexture;
        image.material.shader = Shader.Find("UI/Unlit/Detail");
    }

    void initSprites() {
        GameObject canvas = GameObject.Find("Canvas");

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

    void Draw3dPoints(TensorFloat joints_tensor) {
        float[] joints = joints_tensor.ToReadOnlyArray();
        int arrlen = joints.Length;

        for (int idx = 0; idx < arrlen; idx += 3) {
            float x = joints[idx] * scale3d;
            float y = joints[idx+1] * scale3d;
            float z = joints[idx+2] * scale3d;

            GameObject point = points3d[idx/3];
            point.transform.localPosition = new Vector3(x, -y, z);
        }
    }
    
    void DrawPoints(TensorFloat joints_tensor) {
        float[] joints = joints_tensor.ToReadOnlyArray();
        int arrlen = joints.Length;

        for (int idx = 0; idx < arrlen; idx += 3) {
            float x = joints[idx];
            float y = joints[idx+1];
            float confidence = joints[idx+2];

            if (x >=0 && x < resizedImageSize && y >= 0 && y < resizedImageSize) {
                DrawPoint(idx, x, y);
            }
        }
    }

    void DrawPoint(int index, float x, float y) {
        GameObject point = points[index/3];
        x = (x-(resizedImageSize/2)) * scaleX;
        y = (y-(resizedImageSize/2)) * scaleY;
        point.transform.localPosition = new Vector3(x, -y, 0);
    }

    void OnDisable() {
        
    }

}
