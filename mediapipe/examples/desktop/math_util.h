namespace m_math{
    //特征点结构体
    struct PoseInfo{
    float x,y,z;
    };
    typedef struct PoseInfo PoseInfo;
    /**
     * 计算两点间欧几里德距离
    */
    float distance(PoseInfo v1,PoseInfo v2);
    /**
     * 计算三点夹角
    */
    float angle(PoseInfo v1,PoseInfo v2, PoseInfo v3);
}
