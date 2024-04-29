using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

public class CreateWorker : MonoBehaviour
{
    ModelAsset modelAsset;
    Model runtimeModel;
    IWorker worker;

    // Start is called before the first frame update
    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
    }

    // Update is called once per frame
    void Update()
    {

    }

    float halpe2h36m(int joints, int scores)
    {
        float n = 2;

        return n;
    }

    void inference_detector()
    {

    }

    void pose2d_predict()
    {

    }

    void inference_topdown()
    {

    }

    void merge_data_samples()
    {

    }

    void pose_3d_predict()
    {

    }

    void action_prediction()
    {

    }

    void nms()
    {

    }
}

