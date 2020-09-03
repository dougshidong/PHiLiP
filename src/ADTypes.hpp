#ifndef __AD_TYPES__
#define __AD_TYPES__

#include <Sacado.hpp>
#include <CoDiPack/include/codi.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

namespace PHiLiP {
using FadType = Sacado::Fad::DFad<double>; ///< Sacado AD type for first derivatives.
using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

static constexpr int dimForwardAD = 1; ///< Size of the forward vector mode for CoDiPack.
static constexpr int dimReverseAD = 1; ///< Size of the reverse vector mode for CoDiPack.

using codi_FadType = codi::RealForwardGen<double, codi::Direction<double,dimForwardAD>>; ///< Tapeless forward mode.
//using codi_FadType = codi::RealForwardGen<double, codi::DirectionVar<double>>;

using codi_JacobianComputationType = codi::RealReverseIndexVec<dimReverseAD>; ///< Reverse mode type for Jacobian computation using TapeHelper.
using codi_HessianComputationType  = codi::RealReversePrimalIndexGen< codi::RealForwardVec<dimForwardAD>,
                                                  codi::Direction< codi::RealForwardVec<dimForwardAD>, dimReverseAD>
                                                >; ///< Nested reverse-forward mode type for Jacobian and Hessian computation using TapeHelper.

//using RadFadType = Sacado::Rad::ADvar<FadType>; ///< Sacado AD type that allows 2nd derivatives.
//using RadFadType = codi_JacobianComputationType; ///< Reverse only mode that only allows Jacobian computation.
using RadType = codi_JacobianComputationType; ///< CoDiPaco reverse-AD type for first derivatives.
using RadFadType = codi_HessianComputationType ; ///< Nested reverse-forward mode type for Jacobian and Hessian computation using TapeHelper.
} // PHiLiP namespace

#endif
