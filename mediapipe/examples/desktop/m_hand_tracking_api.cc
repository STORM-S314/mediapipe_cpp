// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/examples/desktop/math_util.h"

#define EXPORT_API __attribute__((visibility ("default")))
#define logfile "Assets/Plugins/mLog.log"
extern "C" {
//特征点回调函数
typedef void(*LandmarksCallBack)(int image_index,m_math::PoseInfo* infos, int count);
LandmarksCallBack m_LandmarksCallBackFunc=nullptr;
//输出流轮询器
std::unique_ptr<mediapipe::OutputStreamPoller> poller;

mediapipe::CalculatorGraph graph;

std::ofstream fout;
cv::VideoCapture capture;
		
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "landmarks";
int StatusCodeToInt(absl::StatusCode code) {
  using namespace absl;
  switch (code) {
    case StatusCode::kOk:
      return 0;
    case StatusCode::kCancelled:
      return 1;
    case StatusCode::kUnknown:
      return 2;
    case StatusCode::kInvalidArgument:
      return 3;
    case StatusCode::kDeadlineExceeded:
      return 4;
    case StatusCode::kNotFound:
      return 5;
    case StatusCode::kAlreadyExists:
      return 6;
    case StatusCode::kPermissionDenied:
      return 7;
    case StatusCode::kUnauthenticated:
      return 16;
    case StatusCode::kResourceExhausted:
      return 8;
    case StatusCode::kFailedPrecondition:
      return 9;
    case StatusCode::kAborted:
      return 10;
    case StatusCode::kOutOfRange:
      return 11;
    case StatusCode::kUnimplemented:
      return 12;
    case StatusCode::kInternal:
      return 13;
    case StatusCode::kUnavailable:
      return 14;
    case StatusCode::kDataLoss:
      return 15;
  }
  return 2;
}
//注册特征点回调函数
EXPORT_API void RegisterLandmarksCallBack(LandmarksCallBack func){
  m_LandmarksCallBackFunc=func;
}

//初始化MediaPipeGraph
absl::Status _InitMPPGraph(const char* model_path){

  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(model_path,&calculator_graph_config_contents));
  fout<<calculator_graph_config_contents<<std::endl;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);
  fout<<"ParseTextProtoOrDie"<<std::endl;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  //添加landmarks输出流
  mediapipe::StatusOrPoller sop_landmark=graph.AddOutputStreamPoller(kOutputStream);
  assert(sop_landmark.ok());
  poller= std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop_landmark.value()));
  fout<<"Add Landmarks"<<std::endl;
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  fout<<"Start Run"<<std::endl;
  return absl::OkStatus();
}
//初始化MediaPipeGraph
EXPORT_API  int InitMPPGraph(const char* model_path){
  fout.open(logfile, std::ios::out);
  absl::Status result=_InitMPPGraph(model_path);
  absl::StatusCode code = result.code();
  std::cout<<absl::StatusCodeToString(code)<<std::endl;
  capture.open(0);
  // RET_CHECK(capture.isOpened());
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  capture.set(cv::CAP_PROP_FPS, 60);
  return StatusCodeToInt(code);
}
/**
 * 预测手势
*/
std::string predictGesture(std::vector<m_math::PoseInfo> hand_landmarks){
  float d_3_5 = m_math::distance(hand_landmarks[3],hand_landmarks[5]);
  float d_2_3 = m_math::distance(hand_landmarks[2],hand_landmarks[3]);
  float angle0 = m_math::angle(hand_landmarks[0],hand_landmarks[1],hand_landmarks[2]);
  float angle1 = m_math::angle(hand_landmarks[1],hand_landmarks[2],hand_landmarks[3]);
  float angle2 = m_math::angle(hand_landmarks[2],hand_landmarks[3],hand_landmarks[4]);
  int thumb = 0;
  int index = 0;
  int middle = 0;
  int ring = 0;
  int little = 0;
  if(angle0+angle1+angle2>460 && d_3_5/d_2_3>1.2){
    thumb=1;
  }else{
    thumb=0;
  }
  if(hand_landmarks[8].y<hand_landmarks[7].y && hand_landmarks[8].y<hand_landmarks[6].y){
    index=1;
  }
  else if(hand_landmarks[8].y>hand_landmarks[6].y){
    index=0;
  }else{
    index=-1;
  }
  if(hand_landmarks[12].y<hand_landmarks[11].y && hand_landmarks[12].y<hand_landmarks[10].y){
    middle=1;
  }
  else if(hand_landmarks[12].y>hand_landmarks[10].y){
    middle=0;
  }else{
    middle=-1;
  }
  if(hand_landmarks[16].y<hand_landmarks[15].y && hand_landmarks[16].y<hand_landmarks[14].y){
    ring=1;
  }
  else if(hand_landmarks[16].y>hand_landmarks[14].y){
    ring=0;
  }else{
    ring=-1;
  }
  if(hand_landmarks[20].y<hand_landmarks[19].y && hand_landmarks[20].y<hand_landmarks[18].y){
    little=1;
  }
  else if(hand_landmarks[20].y>hand_landmarks[18].y){
    little=0;
  }else{
    little=-1;
  }
  if(thumb<0||index<0||middle<0||ring<0||little<0){
    return "None";
  }
  if(thumb==1&&index==1&&middle==1&&ring==1&&little==1){
    return "Five";
  }
  if(thumb==1&&index==1&&middle==0&&ring==0&&little==0){
    return "Zoom";
  }
  if(thumb==1&&index==1&&middle==1&&ring==0&&little==0){
    return "Three";
  }
  if(thumb==0&&index==1&&middle==1&&ring==1&&little==0){
    return "Three";
  }
  if(thumb==0&&index==1&&middle==1&&ring==1&&little==1){
    return "Four";
  }
  if(thumb==0&&index==1&&middle==1&&ring==0&&little==0){
    return "Peace";
  }
  if(thumb==0&&index==1&&middle==0&&ring==0&&little==0){
    return "One";
  }
  if(thumb==0&&index==0&&middle==0&&ring==0&&little==0){
    return "Fist";
  }

}
absl::Status RunMPPGraph(int cSock) {
  bool grab_frames=true;
  while (grab_frames) {
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    
      // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);
    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));
    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    
    if(poller->QueueSize()>0){
        if (poller->Next(&packet)){
          
          std::vector<mediapipe::NormalizedLandmarkList> output_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
          std::vector<m_math::PoseInfo> hand_landmarks;
          hand_landmarks.clear();
          for (int m = 0; m < output_landmarks.size(); ++m)
          {
              mediapipe::NormalizedLandmarkList single_hand_NormalizedLandmarkList = output_landmarks[m];
              for (int i = 0; i < single_hand_NormalizedLandmarkList.landmark_size(); ++i)
              {
                  m_math::PoseInfo info;
                  const mediapipe::NormalizedLandmark landmark = single_hand_NormalizedLandmarkList.landmark(i);
                  info.x = landmark.x();
                  info.y = landmark.y();
                  info.z = landmark.z();
                  hand_landmarks.push_back(info);
              }
          }
          //判断手势
          std::string gesture = predictGesture(hand_landmarks);
          //将坐标信息转为str发送到客户端
          std::string hand_landmarks_pose_infos=gesture+"\n";
          for (int i = 0; i < hand_landmarks.size(); ++i)
          {
            hand_landmarks_pose_infos+= std::to_string(i);
            hand_landmarks_pose_infos+= ",";
            
            hand_landmarks_pose_infos+= std::to_string(hand_landmarks[i].x);
            hand_landmarks_pose_infos+= ",";
            hand_landmarks_pose_infos+= std::to_string(hand_landmarks[i].y);
            hand_landmarks_pose_infos+= ",";
            hand_landmarks_pose_infos+= std::to_string(hand_landmarks[i].z);
            hand_landmarks_pose_infos+= "\n";
          }
          const char* msg = hand_landmarks_pose_infos.c_str();
          send(cSock,msg,sizeof(char)*hand_landmarks_pose_infos.length(),0);
          std::cout<<"bytes:"<<sizeof(char)*hand_landmarks_pose_infos.length()<<hand_landmarks_pose_infos<<std::endl;
        }
    }

    
  }

  return absl::OkStatus();
}
EXPORT_API int DetectionFrame(int cSock){
  absl::Status result=RunMPPGraph(cSock);
  absl::StatusCode code = result.code();
  std::cout<<absl::StatusCodeToString(code)<<std::endl;
  return StatusCodeToInt(code);
}

absl::Status _Release(){
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

EXPORT_API int Release(){
  return _Release()==absl::OkStatus()?0:1;
}

}

void callback(int image_index,m_math::PoseInfo* infos, int count){
  std::cout<<"has info"<<std::endl;
}
int main(int argc, char** argv) {
  RegisterLandmarksCallBack(callback);
  int result = InitMPPGraph("mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt");
  std::cout<<"result:"<<result<<std::endl;
  
  int server = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
  if(server==-1){
    std::cout<<"socket init error"<<std::endl;
  }
  sockaddr_in sin = {};
  sin.sin_family=AF_INET;
  sin.sin_port=htons(10000);
  sin.sin_addr.s_addr=INADDR_ANY;
  if(bind(server,(sockaddr*)&sin,sizeof(sin))==-1){
    std::cout<<"socket bind error"<<std::endl;
  }
  if(listen(server,5)==-1){
    std::cout<<"socket listen error"<<std::endl;
  }
  sockaddr_in clientAddr={};
  socklen_t nAddrLen = sizeof(sockaddr_in);
  int cSock = -1;
  while(true){
    cSock = accept(server,(sockaddr*)&clientAddr,&nAddrLen);
    if(cSock==-1){
      std::cout<<"socket accept error"<<std::endl;
    }
    result = DetectionFrame(cSock);
    std::cout<<"result:"<<result<<std::endl;
  }
  close(server);
  return EXIT_SUCCESS;
}