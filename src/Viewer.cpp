//
// Created by mpl on 24-1-4.
//

#include "Viewer.h"
#include <Eigen/Core>

using namespace CannyEVIT;

Viewer::Viewer(const std::string &config_path, System *system) : system_(system) {

  cv::FileStorage fs(config_path, cv::FileStorage::READ);
  LOG(INFO) << "-------------- Viewer --------------";

  if (!fs.isOpened()) LOG(FATAL) << "config not open: " << config_path;

  if (fs["point_size"].isNone()) LOG(ERROR) << "config: point_size is not set";
  point_size_ = fs["point_size"];

  if (fs["frame_size"].isNone()) LOG(ERROR) << "config: frame_size is not set";
  frame_size_ = fs["frame_size"];

  LOG(INFO) << "point_size_:" << point_size_;
  LOG(INFO) << "frame_size_:" << frame_size_;
}

void Viewer::Run() {

  Eigen::MatrixXd points(system_->cloud_->size(), 3);
  points.setZero();
  Eigen::MatrixXd gradients(system_->cloud_->size(), 3);
  gradients.setZero();

  Eigen::Vector3d cloud_center = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < system_->cloud_->size(); i++) {
    Eigen::Vector3d point(system_->cloud_->at(i).x, system_->cloud_->at(i).y, system_->cloud_->at(i).z);
    Eigen::Vector3d gradient
        (system_->cloud_->at(i).x_gradient_, system_->cloud_->at(i).y_gradient_, system_->cloud_->at(i).z_gradient_);

    points.block<1, 3>(i, 0) = point;
    gradients.block<1, 3>(i, 0) = (gradient - point).normalized() * 0.02 + point;
//    gradients.block<1, 3>(i, 0) = gradient;
    cloud_center += point;
  }
  cloud_center /= system_->cloud_->size();

  Eigen::Vector3d init_t = system_->t0_;

  pangolin::CreateWindowAndBind("EVIT Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuDrawGradient("menu.Draw Gradient", false, true);
  pangolin::Var<bool> menuDrawIMU("menu.Draw IMU frame", true, true);
  pangolin::Var<bool> menuDrawEvent("menu.Draw Event frame", true, true);

  pangolin::Var<bool> menuStepByStep("menu.Step By Step", false, true);  // false, true
  pangolin::Var<bool> menuStep("menu.Step", false, false);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500.0, 500.0, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(init_t.x(), init_t.y(), init_t.z(),
                                cloud_center.x(), cloud_center.y(), cloud_center.z(),
                                0.0, 0.0, 1.0)
  );

  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View &d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // draw axis
    glColor3f(1.0, 0.0, 0.0);
    glLineWidth(3);
    pangolin::glDrawLine(cloud_center.x(), cloud_center.y(), cloud_center.z(),
                         cloud_center.x() + 0.2, cloud_center.y(), cloud_center.z());
    glColor3f(0.0, 1.0, 0.0);
    pangolin::glDrawLine(cloud_center.x(), cloud_center.y(), cloud_center.z(),
                         cloud_center.x(), cloud_center.y() + 0.2, cloud_center.z());
    glColor3f(0.0, 0.0, 1.0);
    pangolin::glDrawLine(cloud_center.x(), cloud_center.y(), cloud_center.z(),
                         cloud_center.x(), cloud_center.y(), cloud_center.z() + 0.2);

    // draw point and gradient
    glPointSize(point_size_);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);
    for (size_t i = 0; i < system_->cloud_->size(); i++) {
      glVertex3d(points(i, 0), points(i, 1), points(i, 2));
    }
    glEnd();

    // draw gradient
    if (menuDrawGradient) {
      glLineWidth(1);
      glBegin(GL_LINES);
      glColor3f(0.0, 1.0, 0.0);

      for (size_t i = 0; i < system_->cloud_->size(); i++) {
        glVertex3d(points(i, 0), points(i, 1), points(i, 2));
        glVertex3d(gradients(i, 0), gradients(i, 1), gradients(i, 2));
      }
      glEnd();
    }

    // draw old frames
    std::vector<Frame::Ptr> frames = system_->getAllFrames();
    for (const auto& frame: frames){
      if (menuDrawIMU)
        drawAxis(frame->Twb());
      if (menuDrawEvent)
        drawCamera(frame->Twc(), frame->isActive());
    }

    pangolin::FinishFrame();
  }

}

void Viewer::drawCamera(const Eigen::Matrix4d &Twc, bool isActivate) {
  const float &w = frame_size_;
  const float h = w * 0.75;
  const float z = w * 0.6;

  glPushMatrix();
  glMultMatrixd((GLdouble *) Twc.data());

  glLineWidth(3.0);
  if (isActivate)
    glColor3f(1.0f, 0.0f, 0.0f);
  else
    glColor3f(0.0f, 0.0f, 1.0f);

  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(w, h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, -h, z);
  glVertex3f(0, 0, 0);
  glVertex3f(-w, h, z);

  glVertex3f(w, h, z);
  glVertex3f(w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(-w, -h, z);

  glVertex3f(-w, h, z);
  glVertex3f(w, h, z);

  glVertex3f(-w, -h, z);
  glVertex3f(w, -h, z);
  glEnd();
  glPopMatrix();
}

void Viewer::drawAxis(const Eigen::Matrix4d &Twb) {
  const float &w = frame_size_;
  glPushMatrix();
  glMultMatrixd((GLdouble *) Twb.data());
  glBegin(GL_LINES);
  glColor3f(1.0, 0.0, 0.0);
  glVertex3f(0, 0, 0);
  glVertex3f(w, 0, 0);
  glColor3f(0.0, 1.0, 0.0);
  glVertex3f(0, 0, 0);
  glVertex3f(0, w, 0);
  glColor3f(0.0, 0.0, 1.0);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 0, w);
  glEnd();
  glPopMatrix();
}


