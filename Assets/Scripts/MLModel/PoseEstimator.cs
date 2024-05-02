using UnityEngine;
using Unity.Sentis;
using FF = Unity.Sentis.Functional;
using Unity.VisualScripting;
using UnityEngine.UI;
using System.Collections.Generic;
using System;

public class PoseEstimator {

    private int webcamWidth;
    private int webcamHeight;
    private float requestedFPS;
    const int resizedImageSize=320;

    private float scaleX;
    private float scaleY;
    private TextureTransform textureTransform;

    private float iouThreshold=0.5f;
    private float scoreThreshold=0.5f;

    private IWorker TwoDPoseWorker;
    private IWorker ThreeDPoseWorker;
    private BackendType backend;
    private TensorFloat inputTensor;
    public PoseEstimator(WebCamTexture webcamTexture, ref ModelAsset TwoDPoseModelAsset, ref ModelAsset ThreeDPoseModelAsset, BackendType backend) {
        this.webcamWidth = webcamTexture.width;
        this.webcamHeight = webcamTexture.height;
        this.requestedFPS = webcamTexture.requestedFPS;

        this.backend = backend;

        scaleX = (float)this.webcamWidth / resizedImageSize;
        scaleY = (float)this.webcamHeight / resizedImageSize;

        textureTransform = new TextureTransform().SetDimensions(width: resizedImageSize, height: resizedImageSize, channels: 3);

        LoadModel(resizedImageSize, ref TwoDPoseModelAsset, ref ThreeDPoseModelAsset);

    }

    private void LoadModel(int imageSize, ref ModelAsset TwoDPoseModelAsset, ref ModelAsset ThreeDPoseModelAsset) {
        var yoloPoseModel = ModelLoader.Load(TwoDPoseModelAsset);
        var motionbertModel = ModelLoader.Load(ThreeDPoseModelAsset);

        TensorFloat centersToCorners = new TensorFloat(new TensorShape(4, 4), new float[]{
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

        this.TwoDPoseWorker = WorkerFactory.CreateWorker(this.backend, yoloPoseModel);
        this.ThreeDPoseWorker = WorkerFactory.CreateWorker(this.backend, motionbertModel);
    }

    public void executeTwoDPoseWorker(WebCamTexture webcamTexture) {
        inputTensor = TextureConverter.ToTensor(webcamTexture, textureTransform);

        Debug.Log("TwoDPose: " + inputTensor.shape);

        this.TwoDPoseWorker.Execute(inputTensor);

        inputTensor.Dispose();
    }

    public TensorFloat getTwoDPoseOutput() {
        TensorFloat twoDJoints = this.TwoDPoseWorker.PeekOutput("output_1") as TensorFloat;
        
        twoDJoints.CompleteOperationsAndDownload();

        return twoDJoints;        
    }

    public void executeThreeDPoseWorker(TensorFloat twoDPoseTensor) {
        this.ThreeDPoseWorker.Execute(twoDPoseTensor);
    }

    public TensorFloat getThreeDPoseOutput() {
        TensorFloat threeDJoints = this.ThreeDPoseWorker.PeekOutput("output_0") as TensorFloat;

        threeDJoints.CompleteOperationsAndDownload();

        return threeDJoints;
    }

    public (float, float) getImageScale() {
        return (this.scaleX, this.scaleY);
    }

    ~PoseEstimator() {
        TwoDPoseWorker.Dispose();
        ThreeDPoseWorker.Dispose();
    }
}