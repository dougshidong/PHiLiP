#ifndef __AD_TYPES__
#define __AD_TYPES__

#include <Sacado.hpp>
#include <CoDiPack/include/codi.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

namespace PHiLiP {
static constexpr int dimForwardAD = 1;
static constexpr int dimReverseAD = 1;
using FadType = Sacado::Fad::DFad<double>; ///< Sacado AD type for first derivatives.
using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.
using HessType = codi::RealReversePrimalIndexGen< codi::RealForwardVec<dimForwardAD>,
                                                  codi::Direction< codi::RealForwardVec<dimForwardAD>, dimReverseAD>
                                                >;
//using RadFadType = Sacado::Rad::ADvar<FadType>; ///< Sacado AD type that allows 2nd derivatives.
using RadFadType = HessType; ///< Sacado AD type that allows 2nd derivatives.
} // PHiLiP namespace

#endif
