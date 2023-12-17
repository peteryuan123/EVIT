//
// Created by mpl on 23-11-17.
//

#ifndef CANNYEVIT_GENERICFUNCTOR_H
#define CANNYEVIT_GENERICFUNCTOR_H

#include <Eigen/Eigen>

namespace CannyEVIT {

template <typename ScalarT, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct GenericFunctor {
  /** undocumented */
  typedef ScalarT Scalar;
  /** undocumented */
  enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };
  /** undocumented */
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  /** undocumented */
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  /** undocumented */
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

  /** undocumented */
  const int m_inputs;
  /** undocumented */
  int m_values;

  /** undocumented */
  GenericFunctor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}

  /** undocumented */
  GenericFunctor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  /** undocumented */
  int inputs() const { return m_inputs; }

  /** undocumented */
  int values() const { return m_values; }

  //    void resetNumberInputs(int inputs) { m_inputs = inputs; }

  void resetNumberValues(int values) { m_values = values; }
};

}  // namespace CannyEVIT

#endif  // CANNYEVIT_GENERICFUNCTOR_H
