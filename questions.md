joint 3 trial:

-0.46 lowest joint position maybe?




fixed joints for the grabbing:

joint 4 -1.48
joint 5 -1,49
joint 6 0.901


# Robot Arm Test Points

Based on the robot's workspace and specifications, here are ten points in 3D space that can be used to test the robot arm's movements:

1. **Maximum Reach, Mid-Height**:
   - `(128.4, 0, 150)`

2. **Maximum Reach, Maximum Height**:
   - `(90.76, 90.76, 200)`

3. **Maximum Reach, Minimum Height**:
   - `(0, 128.4, 50)`

4. **Minimum Reach, Mid-Height**:
   - `(-20, -20, 150)`
   - Assuming a minimum reach of 28.28 (which is 20% of the maximum radius).

5. **Directly Above Base, Maximum Height**:
   - `(0, 0, 200)`

6. **Directly Above Base, Minimum Height**:
   - `(0, 0, 50)`

7. **Halfway Maximum Reach, Maximum Height**:
   - `(-45.38, -45.38, 200)`
   - Halfway between center and maximum radial distance.

8. **Halfway Maximum Reach, Minimum Height**:
   - `(-45.38, 45.38, 50)`

9. **Three-Quarter Maximum Reach, Mid-Height**:
   - `(22.63, -99.52, 150)`
   - Three quarters out towards maximum reach.

10. **Edge of Reach, Lower Height**:
    - `(128.39, -1.75, 100)`
    - Just before completing a full circle.

Adjust these values based on your actual robot's reach and height capabilities. Ensure that all dimensions and angles conform to the real mechanical limits of the robot.


### checkpoints

#### 0707 

新的角度计算已经完成了，更改控制的方程，pid无论如何要调试完成一次，得到一个差不多的控制方式

先从9.8重力开始，试一下前两个条件的控制，记得修改pid到一个统一的条件控制，不单独控制一个关节，使用多个控制的点来进行控制

#### 0708
slides making

调整了机械臂给的力度，不确定是否完善
控制的部分还有不断的小幅度震荡，pid继续调整

实际的坐标反馈还有问题，xyz的对应和实际机械臂的坐标对应不上，重新看kinematic的部分