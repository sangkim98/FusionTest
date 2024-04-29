using UnityEngine;
using Unity.Sentis;
using Lays = Unity.Sentis.Layers;
using FF = Unity.Sentis.Functional;
using System.Collections;
using Unity.VisualScripting;
using UnityEngine.UI;
using System.Collections.Generic;
using System;
using UnityEditor;
using System.Net;

public class PoseEstimator : MonoBehaviour
{
    // Game Objects
    private List<GameObject> points;
    private GameObject poseImage;
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

        initSprites();
        initPoseImage();

        LoadModel();
    }

    void Update()
    {
        inputTensor = TextureConverter.ToTensor(webcamTexture, textureTransform);
        
        yoloPoseWorker.Execute(inputTensor);

        TensorFloat joints_yolo = yoloPoseWorker.PeekOutput("output_0") as TensorFloat;
        TensorFloat joints_h36m = yoloPoseWorker.PeekOutput("output_1") as TensorFloat;

        joints_yolo.CompleteOperationsAndDownload();
        joints_h36m.CompleteOperationsAndDownload();

        joints_h36m.PrintDataPart(17, "Data: ");

        DrawPoints(joints_yolo);

        inputTensor.Dispose();
    }

    void LoadModel() {
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
                var x = joints_coords.Reshape(new int[] { 1, 1, 3, -1 }).Transpose(2,3);    // shape=(1,1,17,3)
                var y = x.Clone();
                y[..,..,0,..] = FF.Add(x[..,..,11,..], x[..,..,12,..]) * 0.5f;
                y[..,..,1,..] = x[..,..,12,..];
                y[..,..,2,..] = x[..,..,14,..];
                y[..,..,3,..] = x[..,..,16,..];
                y[..,..,4,..] = x[..,..,11,..];
                y[..,..,5,..] = x[..,..,13,..];
                y[..,..,6,..] = x[..,..,15,..];
                y[..,..,8,..] = FF.Add(x[..,..,5,..], x[..,..,6,..]) * 0.5f;
                y[..,..,7,..] = FF.Add(y[..,..,0,..], y[..,..,8,..]) * 0.5f;
                y[..,..,9,..] = x[..,..,0,..];
                y[..,..,10,..] = FF.Add(x[..,..,1,..], x[..,..,2,..]) * 0.5f;
                y[..,..,11,..] = x[..,..,5,..];
                y[..,..,12,..] = x[..,..,7,..];
                y[..,..,13,..] = x[..,..,9,..];
                y[..,..,14,..] = x[..,..,6,..];
                y[..,..,15,..] = x[..,..,8,..];
                y[..,..,16,..] = x[..,..,10,..];
                return (joints_coords,y.Transpose(2,3));
            },
            InputDef.FromModel(yoloPoseModel)[0]
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

    void skeletonRenderer() {
        
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

    void coco2h36m(FunctionalTensor x) {
        /*
            Input: x (M x T x V x C)
            
            COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
            
            H36M:
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        */


    }
}
