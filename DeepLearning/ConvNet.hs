{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE FlexibleInstances         #-}
{-# LANGUAGE RecordWildCards           #-}
{-# LANGUAGE TypeOperators             #-}

{-|
Module      : DeepLearning.ConvNet
Description : Deep Learning
Copyright   : (c) Andrew Tulloch, 2014
License     : GPL-3
Maintainer  : andrew+cabal@tullo.ch
Stability   : experimental
Portability : POSIX
-}
module DeepLearning.ConvNet
    -- (
    --  -- ** Main Types
    --  Vol,
    --  DVol,
    --  Label,
    --  -- ** Layers
    --  -- Layer,
    --  -- InnerLayer,
    --  -- TopLayer,
    --  -- SoftMaxLayer(..),
    --  -- FullyConnectedLayer(..),
    --  -- forward,
    --  -- ** Composing layers
    --  -- (>->),
    --  -- Forward,
    --  -- withActivations,

    --  -- ** Network building helpers
    --  -- flowNetwork,
    --  -- net1,
    --  -- net2,
    --  -- newFC,
    -- )
    where

import           Data.Array.Repa hiding (toList)
import qualified Data.Array.Repa as R
import qualified Data.List       as L
import           Prelude         hiding (map, zipWith)
import qualified Prelude         as P

data SGD = Vanilla { _learningRate :: Double }
data Network a = Network { _innerLayers :: [a], _topLayer :: a }
data Example = Example { _input :: DVol DIM1, _label :: Label }

type DVol sh = Array D sh Double

data Activations = Activations {
      _inputAct  :: DVol DIM1,
      _outputAct :: DVol DIM1
    }

data Gradients = Gradients { _layerGrads :: [(Layer, [DVol DIM1])]}
data BackProp = BackProp { _paramGrad :: [DVol DIM1], _inputGrad :: DVol DIM1 }
data InputProp = InputProp {
      _outputGrad  :: DVol DIM1,
      _activations :: Activations
    }

-- |Label for supervised learning
type Label = Int

-- |'Layer' reprsents a layer that can pass activations forward and backward
data Layer = Layer {
      _forward       :: DVol DIM1 -> DVol DIM1,
      _topBackward   :: Label -> Activations -> BackProp,
      _innerBackward :: InputProp -> BackProp,
      _applyGradient :: SGD -> [BackProp] -> Layer
    }

-- |'SoftMaxLayer' computes the softmax activation function.
softMaxLayer :: Layer
softMaxLayer = Layer {
                 _forward = softMaxForward,
                 _topBackward = softMaxBackward,
                 _innerBackward = error "Should not be called on SoftMaxLayer",
                 _applyGradient = \_ _ -> softMaxLayer
               }

softMaxForward :: (Shape sh) => DVol sh -> DVol sh
softMaxForward input = w where
    exponentials = exponentiate input
    sumE = foldAllS (+) 0.0 exponentials
    w = map (/ sumE) exponentials
    maxA = foldAllS max 0.0
    exponentiate acts = map (\a -> exp (a - maxAct)) acts
        where
          maxAct = maxA acts

softMaxBackward :: Label -> Activations -> BackProp
softMaxBackward label Activations{..} = BackProp undefined (R.traverse _outputAct id gradientAt)
      where
        gradientAt f s@(Z :. i) = gradient (f s) i
        gradient outA target = -(bool2Double indicator - outA)
            where
              indicator = label == target
              bool2Double x = if x then 1.0 else 0.0

-- |'FullyConnectedLayer' represents a fully-connected input layer
data FullyConnectedState = FullyConnectedState {
      _bias    :: [Double],
      _weights :: [DVol DIM1]
    }

fcLayer :: FullyConnectedState -> Layer
fcLayer fcState = Layer {
                    _forward = fcForward fcState,
                    _innerBackward = fcBackward fcState,
                    _topBackward = error "Should not be called",
                    _applyGradient = fcApplyGradient fcState
                  }

fcApplyGradient :: FullyConnectedState -> SGD -> [BackProp] -> Layer
fcApplyGradient fcState sgd (biasG:weightsG) = fcLayer newState
    where
      newState = undefined

fcForward :: FullyConnectedState -> DVol DIM1 -> DVol DIM1
fcForward FullyConnectedState{..} input = delay $ fromListUnboxed (Z :. numFilters) outputs
        where
          numFilters = length _bias
          output :: Double -> DVol DIM1 -> Double
          output b w = b + dotProduct input w
          outputs = P.zipWith output _bias _weights
          dotProduct a b = sumAllS $ zipWith (+) a b

fcBackward :: FullyConnectedState -> InputProp -> BackProp
fcBackward = undefined

applyForward :: Network t -> a -> (a -> t -> a) -> Network (t, a)
applyForward Network{..} input f = Network newInner newTop
    where
     innerActs = tail $ scanl f input _innerLayers
     newInner = zip _innerLayers innerActs
     topAct = f (last innerActs) _topLayer
     newTop = (_topLayer, topAct)

activations
  :: Network Layer -> Activations -> Network (Layer, Activations)
activations net input = applyForward net input layerActs where
    layerActs Activations{..} layer = Activations _outputAct (_forward layer _outputAct)

instance Functor Network where
    fmap f Network{..} = Network (fmap f _innerLayers) (f _topLayer)

-- instance Foldable t => Foldable (Network t) where
--     foldMap

netToList :: Network a -> [a]
netToList Network{..} = _innerLayers P.++ [_topLayer]

netFromList :: [a] -> Network a
netFromList layers = Network (init layers) (last layers)

applyBackward
  :: Network (t1, b)
     -> t
     -> (t -> (t1, b) -> t2)
     -> (t2 -> (t1, b) -> t2)
     -> Network (t1, t2)
applyBackward Network{..} topInput topF innerF = Network newInner newTop
    where
      topBackward = topF topInput _topLayer
      newTop = (fst _topLayer, topBackward)
      innerBackward = scanl innerF topBackward (reverse _innerLayers)
      newInner = zip (P.fmap fst _innerLayers) (reverse innerBackward)

backprop :: Network (Layer, Activations) -> Label -> Network (Layer, BackProp)
backprop net label = applyBackward net label topBackward innerBackward
    where
      topBackward :: Label -> (Layer, Activations) -> BackProp
      topBackward label (layer, output) = _topBackward layer label output
      innerBackward :: BackProp -> (Layer, Activations) -> BackProp
      innerBackward BackProp{..} (layer, acts) = _innerBackward layer $ InputProp _inputGrad acts

exampleGradients
  :: Network Layer -> (DVol DIM1, Label) -> Network (Layer, BackProp)
exampleGradients net (input, label) = backprop activated label
    where
      -- FIXME - this isn't a great pattern...
      activated = activations net (Activations undefined input)

batchGradients :: Network Layer -> [(DVol DIM1, Label)] -> Network (Layer, [BackProp])
batchGradients net examples = netFromList (zip (netToList net) mergedGrads)
    where
      gradients :: [[(Layer, BackProp)]]
      gradients = P.fmap (netToList . exampleGradients net) examples
      layerGrads = L.transpose gradients
      mergedGrads = P.fmap (P.fmap snd) layerGrads

applyGradients :: Network (Layer, [BackProp]) -> SGD -> Network Layer
applyGradients net sgd = P.fmap (\(layer, backprops) -> _applyGradient layer sgd backprops) net

runBatch :: Network Layer -> SGD -> [(DVol DIM1, Label)] -> Network Layer
runBatch net sgd examples = (`applyGradients` sgd) $ batchGradients net examples
