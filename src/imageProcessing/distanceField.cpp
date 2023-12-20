#include "imageProcessing/distanceField.h"
#include "imageProcessing/image_processing.h"
#include <thread>
using namespace CannyEVIT;

void image_processing::chebychevDistanceField(size_t rows,
                                              size_t cols,
                                              const std::vector<std::pair<int, int>> &uv_edge,
                                              Eigen::ArrayXXd &df,
                                              size_t radius) {

  if (df.rows() != rows || df.cols() != cols)
    df.resize(rows, cols);
  df.fill((double) radius);

  std::vector<std::pair<int, int> > *neighbours = new std::vector<std::pair<int, int> >();
  neighbours->reserve(8 * uv_edge.size()); //because in the worst case every point has 8 neighbours

  for (const auto &uv : uv_edge) {
    size_t row = uv.first;
    size_t col = uv.second;
    neighbours->emplace_back(row, col);
    df(row, col) = 0.01f;
  }

  double distance = 1.01f;
  while (!neighbours->empty()) {
    std::vector<std::pair<int, int> > *newNeighbours = new std::vector<std::pair<int, int> >();
    newNeighbours->reserve(8 * neighbours->size());

    for (auto &currentNeighbour : *neighbours) {
      //now check the neighbours, set them if required (value bigger than (distance), and store their location
      size_t row = currentNeighbour.first;
      size_t col = currentNeighbour.second;

      if (row > 0 && col > 0 && row < rows - 1 && col < cols - 1) {
        for (size_t r = row - 1; r <= row + 1; r++) {
          for (size_t c = col - 1; c <= col + 1; c++) {
            if (df(r, c) > distance) {
              df(r, c) = distance;
              newNeighbours->emplace_back(r, c);
            }
          }
        }
      }
    }

    delete neighbours;
    neighbours = newNeighbours;
    distance += 1.0f;
  }
  delete neighbours;
}

void image_processing::chamferDistanceField(size_t rows,
                                            size_t cols,
                                            const std::vector<std::pair<int, int>> &uv_edge,
                                            Eigen::ArrayXXd &df,
                                            size_t radius) {

  if (df.rows() != rows || df.cols() != cols)
    df.resize(rows, cols);
  //df.fill( (double) sqrt(rows*rows+cols*cols) );
  df.fill((double) radius);

  for (const auto &uv : uv_edge) {
    df(uv.first, uv.second) = 0.0f;
  }

  std::vector<double> LD; // Local Distance Metric
  std::vector<double> DX; // Mask Array
  std::vector<double> DY; // Mask Array

  // define chamfer values
  double a1 = 2.2062f;
  double a2 = 1.4141f;
  double a3 = 0.9866f;

  // forward scan
  LD.push_back(a1);
  LD.push_back(a1);
  LD.push_back(a1);
  LD.push_back(a2);
  LD.push_back(a3);
  LD.push_back(a2);
  LD.push_back(a1);
  LD.push_back(a3);
  DX.push_back(-2);
  DX.push_back(-2);
  DX.push_back(-1);
  DX.push_back(-1);
  DX.push_back(-1);
  DX.push_back(-1);
  DX.push_back(-1);
  DX.push_back(0);
  DY.push_back(-1);
  DY.push_back(1);
  DY.push_back(-2);
  DY.push_back(-1);
  DY.push_back(0);
  DY.push_back(1);
  DY.push_back(2);
  DY.push_back(-1);

  for (size_t r = 2; r < rows - 2; r++) {
    for (size_t c = 2; c < cols - 2; c++) {
      double d0 = df(r, c);

      for (size_t k = 0; k < 8; k++) {
        size_t r2 = r + DX[k];
        size_t c2 = c + DY[k];

        double d = df(r2, c2) + LD[k];
        if (d < d0)
          d0 = d;
      }

      df(r, c) = d0;
    }
  }

  // backward scan
  LD.clear();
  DX.clear();
  DY.clear();
  LD.push_back(a3);
  LD.push_back(a1);
  LD.push_back(a2);
  LD.push_back(a3);
  LD.push_back(a2);
  LD.push_back(a1);
  LD.push_back(a1);
  LD.push_back(a1);
  DX.push_back(0);
  DX.push_back(1);
  DX.push_back(1);
  DX.push_back(1);
  DX.push_back(1);
  DX.push_back(1);
  DX.push_back(2);
  DX.push_back(2);
  DY.push_back(1);
  DY.push_back(-2);
  DY.push_back(-1);
  DY.push_back(0);
  DY.push_back(1);
  DY.push_back(2);
  DY.push_back(-1);
  DY.push_back(1);

  for (size_t r = rows - 3; r >= 2; r--) {
    for (size_t c = cols - 3; c >= 2; c--) {
      double d0 = df(r, c);

      for (size_t k = 0; k < 8; k++) {
        size_t r2 = r + DX[k];
        size_t c2 = c + DY[k];

        double d = df(r2, c2) + LD[k];
        if (d < d0)
          d0 = d;
      }

      df(r, c) = d0;
    }
  }
}

void image_processing::euclideanDistanceField(size_t rows,
                                              size_t cols,
                                              const std::vector<std::pair<int, int>> &uv_edge,
                                              Eigen::ArrayXXd &df,
                                              size_t radius) {
  typedef Eigen::MatrixXi IndexMap;

  std::vector<std::pair<int, int> > n8;
  n8.reserve(8);
  n8.emplace_back(-1, -1);
  n8.emplace_back(-1, 0);
  n8.emplace_back(-1, 1);
  n8.emplace_back(0, -1);
  n8.emplace_back(0, 1);
  n8.emplace_back(1, -1);
  n8.emplace_back(1, 0);
  n8.emplace_back(1, 1);

  //Astute: create a look-up table for Euclidean distances, as there is only a limited number of options
  Eigen::MatrixXd distanceLUT(radius + 3, radius + 3);
  for (size_t r = 0; r < (radius + 3); r++) {
    for (size_t c = 0; c < (radius + 3); c++)
      distanceLUT(r, c) = sqrt(r * r + c * c);
  }


  //initialize the distance field
  if (df.rows() != rows || df.cols() != cols)
    df.resize(rows, cols);
  df.fill((double) radius);

  //extract first neighbourhood, the seeds themselves
  std::vector<TiedPoint> *neighbours = new std::vector<TiedPoint>();
  neighbours->reserve(uv_edge.size());

  for (const auto &uv : uv_edge) {
    size_t row = uv.first;
    size_t col = uv.second;
    neighbours->emplace_back(row, col, row, col);
    df(row, col) = 0.0f;
  }

  IndexMap indexMap(rows, cols);
  indexMap.fill(-1);

  while (!neighbours->empty()) {
    std::vector<TiedPoint> *newNeighbours = new std::vector<TiedPoint>();
    newNeighbours->reserve(8 * neighbours->size());

    //find all new neighbours where we can have an improvement (ensuring their uniqueness)
    auto it = neighbours->begin();
    while (it != neighbours->end()) {
      for (size_t k = 0; k < n8.size(); k++) {
        int r = it->r + n8[k].first;
        int c = it->c + n8[k].second;

        if (r >= 0 && c >= 0 && r < rows && c < cols) {
          double distance = distanceLUT(abs(r - it->r_origin), abs(c - it->c_origin));
          //TODO: insert some epsilon here, not to risk jumping back and forth
          if (distance < df(r, c)) {
            df(r, c) = distance;

            if (indexMap(r, c) == -1) {
              newNeighbours->emplace_back(r, c, it->r_origin, it->c_origin);
              indexMap(r, c) = newNeighbours->size() - 1;
            } else {
              TiedPoint &tp = (*newNeighbours)[indexMap(r, c)];
              tp.r_origin = it->r_origin;
              tp.c_origin = it->c_origin;
            }
          }
        }
      }

      it++;
    }

    //now go through all the new neighbours, and reset the indices in the index map
    it = newNeighbours->begin();
    while (it != newNeighbours->end()) {
      TiedPoint &tp = *it;
      indexMap(tp.r, tp.c) = -1;
      it++;
    }

    delete neighbours;
    neighbours = newNeighbours;
  }

  delete neighbours;

}

void image_processing::orientedEuclideanDistanceFields(size_t rows,
                                                       size_t cols,
                                                       const std::vector<std::pair<int, int>> &uv_edge,
                                                       const Eigen::ArrayXXd &grad_x,
                                                       const Eigen::ArrayXXd &grad_y,
                                                       std::vector<Eigen::ArrayXXd> &dfs,
                                                       size_t radius) {
  std::vector<std::pair<int, int> > n8;
  n8.reserve(8);
  n8.emplace_back(-1, -1);
  n8.emplace_back(-1, 0);
  n8.emplace_back(-1, 1);
  n8.emplace_back(0, -1);
  n8.emplace_back(0, 1);
  n8.emplace_back(1, -1);
  n8.emplace_back(1, 0);
  n8.emplace_back(1, 1);

  //Astute: create a look-up table for Euclidean distances, as there is only a limited number of options
  size_t halfWidth = (radius + 2);
  size_t width = (halfWidth + 1);
  std::vector<double> distanceLUT(width * width);
  for (size_t r = 0; r <= halfWidth; r++) {
    for (size_t c = 0; c <= halfWidth; c++)
      distanceLUT[r * width + c] = sqrt((double) (r * r + c * c));
  }


  //initialize 8 neighbourhood vectors and dfs
  std::vector<std::vector<TiedPoint> *> allNeighbours;
  allNeighbours.reserve(8);
  for (size_t i = 0; i < 8; i++) {
    allNeighbours.push_back(new std::vector<TiedPoint>());
    allNeighbours.back()->reserve(uv_edge.size());
  }

  if (dfs.size() != 8)
    dfs.resize(8);

  //go through the depth map and fill those vectors depending on orientation
  for (const auto& uv: uv_edge){
    size_t r = uv.first;
    size_t c = uv.second;
    if (r > radius && c > radius && r < (rows - 1 - radius) && c < (cols - 1 - radius)) {
      //compute the orientation
      Eigen::Vector2f v(grad_x(r, c), grad_y(r, c));
      size_t orient = computeOrientationBin(v);
      allNeighbours[orient]->emplace_back(r, c, r, c);
    }
  }

  std::vector<DFjob> jobs(8);
  for (size_t i = 0; i < 8; i++) {
    jobs[i]._df = &dfs[i];
    jobs[i]._neighbours = allNeighbours[i];
    jobs[i]._n8 = &n8;
    jobs[i]._distanceLUT = &distanceLUT;
    jobs[i]._width = width;
    jobs[i]._rows = rows;
    jobs[i]._cols = cols;
    jobs[i]._radius = radius;
  }

  std::vector<std::thread> dfComputers;
  dfComputers.reserve(8);
  for (size_t i = 0; i < 8; i++)
    dfComputers.emplace_back(std::bind(orientedEuclideanDistanceFields_thread, jobs[i]));

  for (auto &thread : dfComputers) {
    if (thread.joinable())
      thread.join();
  }

}

void image_processing::orientedEuclideanDistanceFields_thread(CannyEVIT::image_processing::DFjob &job) {
  typedef Eigen::MatrixXi IndexMap;

  //initialize references
  Eigen::ArrayXXd &df = *(job._df);
  std::vector<TiedPoint> *neighbours = job._neighbours;
  const std::vector<std::pair<int, int> > &n8 = *(job._n8);
  const std::vector<double> &distanceLUT = *(job._distanceLUT);
  size_t width = job._width;
  size_t rows = job._rows;
  size_t cols = job._cols;
  size_t radius = job._radius;

  //initialize the distance field
  if (df.rows() != rows || df.cols() != cols)
    df.resize(rows, cols);
  df.fill((double) radius);

  IndexMap indexMap(rows, cols);
  indexMap.fill(-1);

  //extract first neighbourhood, the seeds themselves
  std::vector<TiedPoint>::iterator it = neighbours->begin();
  while (it != neighbours->end()) {
    size_t row = it->r;
    size_t col = it->c;
    df(row, col) = 0.0f;
    indexMap(row, col) = -2;
    it++;
  }

  for (size_t q = 0; q < radius; q++) {
    std::vector<TiedPoint> *newNeighbours = new std::vector<TiedPoint>();
    newNeighbours->reserve(3 * neighbours->size());

    std::vector<TiedPoint> &neighboursRef = *neighbours;
    std::vector<TiedPoint> &newNeighboursRef = *newNeighbours;

    //find all new neighbours where we can have an improvement (ensuring their uniqueness)
    for (size_t i = 0; i < neighboursRef.size(); i++) {
      TiedPoint &tp = neighboursRef[i];

      for (size_t k = 0; k < n8.size(); k++) {
        int r = tp.r + n8[k].first;
        int c = tp.c + n8[k].second;

        if (indexMap(r, c) > -1) {
          double distance = distanceLUT[abs(r - tp.r_origin) * width + abs(c - tp.c_origin)];
          if (distance < df(r, c)) {
            df(r, c) = distance;
            TiedPoint &old_tp = newNeighboursRef[indexMap(r, c)];
            old_tp.r_origin = tp.r_origin;
            old_tp.c_origin = tp.c_origin;
          }
        } else {
          if (indexMap(r, c) == -1) {
            double distance = distanceLUT[abs(r - tp.r_origin) * width + abs(c - tp.c_origin)];
            if (distance < df(r, c)) {
              df(r, c) = distance;
              newNeighboursRef.emplace_back(r, c, tp.r_origin, tp.c_origin);
              indexMap(r, c) = newNeighboursRef.size() - 1;
            }
          }
        }
      }
    }

    //now go through all the new neighbours, and reset the indices in the index map
    it = newNeighboursRef.begin();
    while (it != newNeighboursRef.end()) {
      TiedPoint &tp = *it;
      indexMap(tp.r, tp.c) = -1;
      it++;
    }

    delete neighbours;
    neighbours = newNeighbours;
  }
  delete neighbours;
}