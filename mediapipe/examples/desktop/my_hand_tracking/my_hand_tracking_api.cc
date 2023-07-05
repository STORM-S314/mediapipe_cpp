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

#define EXPORT_API __attribute__((visibility ("default")))
#define logfile "Assets/Plugins/mLog.log"
extern "C" {
//特征点结构体
struct PoseInfo{
  float x,y;
};
struct Color32
{
    uchar red;
    uchar green;
    uchar blue;
    uchar alpha;
};
typedef struct PoseInfo PoseInfo;
//特征点回调函数
typedef void(*LandmarksCallBack)(int image_index,PoseInfo* infos, int count);
LandmarksCallBack m_LandmarksCallBackFunc=nullptr;
//输出流轮询器
std::unique_ptr<mediapipe::OutputStreamPoller> poller;
std::unique_ptr<mediapipe::OutputStreamPoller> video_poller;

mediapipe::CalculatorGraph graph;

std::ofstream fout;

		
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStreamVideo[] = "output_video";
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
  mediapipe::StatusOrPoller sop_landmark=graph.AddOutputStreamPoller(kOutputStreamVideo);
  assert(sop_landmark.ok());
  video_poller= std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop_landmark.value()));

  sop_landmark=graph.AddOutputStreamPoller(kOutputStream);
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
  return StatusCodeToInt(code);
}

absl::Status RunMPPGraph(int image_index, int image_width, int image_height, Color32** image_data) {
  //构造cv:Mat
  fout<<"image_index:"<<image_index<<"("<<image_width<<","<<image_height<<")";
  cv::Mat camera_frame_raw(cv::Size(image_width,image_height),CV_8UC4,*image_data);
  fout<<" image_raw:"<<"channels"<<camera_frame_raw.channels()<<"("<<camera_frame_raw.size()<<")";
  cv::Mat camera_frame;
  cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_RGBA2RGB);
  cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ -1);
  fout<<" image:"<<"channels"<<camera_frame.channels()<<"("<<camera_frame.size()<<")";
  fout<<std::endl;
  cv::Mat camera_frame_out;
  cv::cvtColor(camera_frame, camera_frame_out, cv::COLOR_RGB2BGR);
  cv::imshow("kWindowName", camera_frame_out);
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
        fout<<"GetLandmark"<<std::endl;
        std::vector<mediapipe::NormalizedLandmarkList> output_landmarks = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        std::vector<PoseInfo> hand_landmarks;
        hand_landmarks.clear();
        for (int m = 0; m < output_landmarks.size(); ++m)
        {
            mediapipe::NormalizedLandmarkList single_hand_NormalizedLandmarkList = output_landmarks[m];
            for (int i = 0; i < single_hand_NormalizedLandmarkList.landmark_size(); ++i)
            {
                PoseInfo info;
                const mediapipe::NormalizedLandmark landmark = single_hand_NormalizedLandmarkList.landmark(i);
                info.x = landmark.x() * camera_frame.cols;
                info.y = landmark.y() * camera_frame.rows;
                hand_landmarks.push_back(info);
            }
        }
        PoseInfo* hand_landmarks_pose_infos = new PoseInfo[hand_landmarks.size()];
        for (int i = 0; i < hand_landmarks.size(); ++i)
        {
            hand_landmarks_pose_infos[i].x = hand_landmarks[i].x;
            hand_landmarks_pose_infos[i].y = hand_landmarks[i].y;
        }
        if (m_LandmarksCallBackFunc)
        {
            m_LandmarksCallBackFunc(image_index, hand_landmarks_pose_infos, hand_landmarks.size());
        }
        delete[] hand_landmarks_pose_infos;
      }
  }
  return absl::OkStatus();
}
EXPORT_API int DetectionFrame(int image_index, int image_width, int image_height, Color32** image_data){
  absl::Status result=RunMPPGraph(image_index, image_width, image_height, image_data);
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

void callback(int image_index,PoseInfo* infos, int count){
  std::cout<<"has info"<<std::endl;
}
int main(int argc, char** argv) {
  RegisterLandmarksCallBack(callback);
  int result = InitMPPGraph("/Users/lijing/Documents/XJTU/media_pipe/mediapipe/mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt");
  std::cout<<"result:"<<result<<std::endl;
  return EXIT_SUCCESS;
}