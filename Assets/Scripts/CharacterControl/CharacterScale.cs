using UnityEngine;
using System.Collections.Generic;
using System;

public class BoneResizer : MonoBehaviour
{

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

    public BoneResizer()
    {

        float rootToBelly = distanceAB(root.position, belly.position);
        float rootToRightHip = distanceAB(root.position, rightHip.position);
        float rootToLeftHip = distanceAB(root.position, leftHip.position);
        float leftHipToLeftKnee = distanceAB(leftHip.position, leftKnee.position);
        float leftKneeToLeftAnkle = distanceAB(leftKnee.position, leftAnkle.position);
        float rightHipToRightKnee = distanceAB(rightHip.position, rightKnee.position);
        float rightKneeToRightAnkle = distanceAB(rightKnee.position, rightAnkle.position);
        float bellyToNeck = distanceAB(belly.position, neck.position);
        float neckToNose = distanceAB(neck.position, nose.position);
        float noseToHead = distanceAB(nose.position, head.position);
        float neckToLeftShoulder = distanceAB(neck.position, leftShoulder.position);
        float leftShoulderToLeftElbow = distanceAB(leftShoulder.position, leftElbow.position);
        float leftElbowToLeftWrist = distanceAB(leftElbow.position, leftWrist.position);
        float neckToRightShoulder = distanceAB(neck.position, rightShoulder.position);
        float rightShoulderToRightElbow = distanceAB(rightShoulder.position, rightElbow.position);
        float rightElbowToRightWrist = distanceAB(rightElbow.position, rightWrist.position);

    }

    private float distanceAB(Vector3 jointA, Vector3 jointB)
    {

        float magnitude = Vector3.Distance(jointA, jointB);

        return magnitude;

    }

    private Vector3 fromAtoB(Vector3 jointA, Vector3 jointB)
    {

        Vector3 aToB = jointB - jointA;

        return aToB;

    }

}