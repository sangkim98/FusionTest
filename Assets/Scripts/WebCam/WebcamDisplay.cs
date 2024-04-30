using UnityEngine;
using Unity.Sentis;
using FF = Unity.Sentis.Functional;
using Unity.VisualScripting;
using UnityEngine.UI;
using System.Collections.Generic;
using System;
using UnityEditor.VersionControl;
using System.Linq.Expressions;

public class PoseEstimator : MonoBehaviour
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
    WebCamTexture webcamTexture;

    // Webcam Setup
    const int webcamWidth = 640;
    const int webcamHeight = 360;

    // Sentis Related
    const int resizedImageSize = 320;
    const BackendType backend = BackendType.GPUCompute;
    public ModelAsset yoloPoseModelAsset;
    public ModelAsset motionbertModelAsset;
    private IWorker yoloPoseWorker;
    private IWorker motionbertWorker;
    private TextureTransform textureTransform;
    
    private TensorFloat inputTensor;
    private TensorFloat centersToCorners;
    private float scaleX;
    private float scaleY;

    [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        for (int i = 0; i < devices.Length; i++)
        {
            print("Webcam available: " + devices[i].name);
        }

        webcamTexture = new WebCamTexture(devices[0].name, webcamWidth, webcamHeight, 60);

        webcamTexture.Play();

        scaleX = (float)webcamWidth / resizedImageSize;
        scaleY = (float)webcamHeight / resizedImageSize;

        textureTransform = new TextureTransform().SetDimensions(width: resizedImageSize, height: resizedImageSize, channels: 3);

        points = new List<GameObject>();
        points3d = new List<GameObject>();

        initSprites();
        init3DKeypoints();
        initPoseImage();

        LoadModel(resizedImageSize);
    }

    void Update()
    {
        inputTensor = TextureConverter.ToTensor(webcamTexture, textureTransform);
        
        yoloPoseWorker.Execute(inputTensor);

        TensorFloat joints_yolo = yoloPoseWorker.PeekOutput("output_0") as TensorFloat;
        TensorFloat joints2d = yoloPoseWorker.PeekOutput("output_1") as TensorFloat;

        joints_yolo.CompleteOperationsAndDownload();
        joints2d.CompleteOperationsAndDownload();

        DrawPoints(joints_yolo);

        motionbertWorker.Execute(joints2d);

        TensorFloat joints_3d = motionbertWorker.PeekOutput("output_0") as TensorFloat;

        joints_3d.CompleteOperationsAndDownload();

        Debug.Log("3D Pose Shape: " + joints_3d.shape);

        Draw3dPoints(joints_3d);

        inputTensor.Dispose();
    }

    void LoadModel(int imageSize) {
        var yoloPoseModel = ModelLoader.Load(yoloPoseModelAsset);
        var motionbertModel = ModelLoader.Load(motionbertModelAsset);

        centersToCorners = new TensorFloat(new TensorShape(4, 4), new float[]{
            1,0,1,0,
            0,1,0,1,
            -0.5f, 0, 0.5f, 0,
            0, -0.5f, 0, 0.5f
        });

        yoloPoseModel = FF.Compile(
            input => {
                var modelOutput = yoloPoseModel.Forward(input)[0];
                var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);                   // shape=(8400,4)
                var jointsCoords = modelOutput[0, 5.., ..].Transpose(0, 1);                 // shape=(8400,51)
                var scores = modelOutput[0, 4, ..] - scoreThreshold;
                var boxCorners = FF.MatMul(boxCoords, FunctionalTensor.FromTensor(centersToCorners));
                var indices = FF.NMS(boxCorners, scores, iouThreshold);                     // shape=(1)
                var indices_joints = indices.Unsqueeze(-1).BroadcastTo(new int[] { 51 });   // shape=(1,51)
                var joints_coords = FF.Gather(jointsCoords, 0, indices_joints);             // shape=(1,51)
                var joints_reshaped = joints_coords.Reshape(new int[] { 1, 1, 17, -1 });    // shape=(1,1,17,3)
                return (joints_coords, joints_reshaped);
            },
            InputDef.FromModel(yoloPoseModel)[0]
        );

        /*
            COCO:
            0: nose 1: Leye 2: Reye 3: Lear 4Rear
            5: Lsho 6: Rsho 7: Lelb 8: Relb 9: Lwri
            10: Rwri 11: Lhip 12: Rhip 13: Lkne 14: Rkne
            15: Lank 16: Rank
            
            H36M:
            0: root, 1: rhip, 2: rkne, 3: rank, 4: lhip,
            5: lkne, 6: lank, 7: belly, 8: neck, 9: nose,
            10: head, 11: lsho, 12: lelb, 13: lwri, 14: rsho,
            15: relb, 16: rwri
        */

        motionbertModel = FF.Compile(
            input => {
                input[..,..,..,..2] -= (float)imageSize / 2;
                input[..,..,..,..2] /= (float)imageSize;
                var y = input.Clone();
                y[..,..,0,..] = (input[..,..,11,..] + input[..,..,12,..]) * 0.5f;
                y[..,..,1,..] = input[..,..,12,..];
                y[..,..,2,..] = input[..,..,14,..];
                y[..,..,3,..] = input[..,..,16,..];
                y[..,..,4,..] = input[..,..,11,..];
                y[..,..,5,..] = input[..,..,13,..];
                y[..,..,6,..] = input[..,..,15,..];
                y[..,..,8,..] = (input[..,..,5,..] + input[..,..,6,..]) * 0.5f;
                y[..,..,7,..] = (y[..,..,0,..] + y[..,..,8,..]) * 0.5f;
                y[..,..,9,..] = input[..,..,0,..];
                y[..,..,10,..] = (input[..,..,1,..] + input[..,..,2,..]) * 0.5f;
                y[..,..,11,..] = input[..,..,5,..];
                y[..,..,12,..] = input[..,..,7,..];
                y[..,..,13,..] = input[..,..,9,..];
                y[..,..,14,..] = input[..,..,6,..];
                y[..,..,15,..] = input[..,..,8,..];
                y[..,..,16,..] = input[..,..,10,..];
                var output = motionbertModel.Forward(y)[0];
                return output;
            },
            InputDef.FromTensor(TensorFloat.AllocNoData(new TensorShape(1,1,17,3)))
        );

        yoloPoseWorker = WorkerFactory.CreateWorker(backend, yoloPoseModel);
        motionbertWorker = WorkerFactory.CreateWorker(backend, motionbertModel);
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
        yoloPoseWorker.Dispose();
        motionbertWorker.Dispose();
    }

}
