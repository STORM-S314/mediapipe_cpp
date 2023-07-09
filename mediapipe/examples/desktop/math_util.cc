#include "math_util.h"
#include "math.h"
namespace m_math{
    float distance(PoseInfo v1,PoseInfo v2){
        return sqrt(pow(v1.x-v2.x,2)+pow(v1.y-v2.y,2)+pow(v1.z-v2.z,2));
    }
    float angle(PoseInfo v1, PoseInfo v2, PoseInfo v3){
        float d12 = distance(v1,v2);
        float d23 = distance(v2,v3);
        PoseInfo v21;
        v21.x=v1.x-v2.x;
        v21.y=v1.y-v2.y;
        v21.z=v1.z-v2.z;
        PoseInfo v23;
        v23.x=v3.x-v2.x;
        v23.y=v3.y-v2.y;
        v23.z=v3.z-v2.z;
        float dot = v21.x*v23.x+v21.y*v23.y+v21.z*v23.z;
        float cos_radian = dot/(d12*d23);
        return acosf(cos_radian);
    }
}