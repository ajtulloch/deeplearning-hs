{-# LANGUAGE FlexibleContexts          #-}
{-# LANGUAGE FlexibleInstances         #-}
{-# LANGUAGE FunctionalDependencies    #-}
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
    (
     -- ** Main Types
     Vol,
     DVol,
     Label,

     -- ** Layers
     Layer,
     InnerLayer,
     TopLayer,
     SoftMaxLayer(..),
     FullyConnectedLayer(..),

     -- ** Composing layers
     (>->),
     Forward,
     withActivations,

     -- ** Network building helpers
     flowNetwork,
     net1,
     net2,
     newFC,
    ) where

import           Control.Monad as CM
import           Control.Monad.Writer                 hiding (Any)
import           Data.Array.Repa
import           Data.Array.Repa.Algorithms.Randomish
import qualified Data.Vector.Unboxed                  as V
import           Prelude                              as P hiding (map, zipWith, traverse)

-- |Activation matrix
type Vol sh = Array U sh Double
-- |Delayed activation matrix
type DVol sh = Array D sh Double

-- |Label for supervised learning
type Label = Int

-- |'Layer' reprsents a layer that can pass activations forward.
-- 'TopLayer' and 'InnerLayer' are derived layers that can be
-- backpropagated through.
class (Shape sh, Shape sh') => Layer a sh sh' | a -> sh, a -> sh' where
    forward :: (Monad m) => a -> Vol sh -> m (DVol sh')

-- |'TopLayer' is a top level layer that can initialize a
-- backpropagation pass.
class Layer a DIM1 DIM1 => TopLayer a where
    topBackward :: (Monad m) => a -> Label -> Vol DIM1 -> Vol DIM1 -> m (DVol DIM1)

-- |'SoftMaxLayer' computes the softmax activation function.
data SoftMaxLayer = SoftMaxLayer --

instance Layer SoftMaxLayer DIM1 DIM1 where
    forward _ = softMaxForward

instance TopLayer SoftMaxLayer where
    topBackward _ = softMaxBackward

softMaxForward :: (Shape sh, Monad m) => Vol sh -> m (DVol sh)
softMaxForward input = do
  exponentials <- exponentiate input
  sumE <- foldAllP (+) 0.0 exponentials
  return $ map (/ sumE) exponentials
      where
        maxA = foldAllP max 0.0
        exponentiate acts = do
              maxAct <- maxA acts
              return $ map (\a -> exp (a - maxAct)) acts

softMaxBackward :: (Monad m) => Label -> Vol DIM1 -> Vol DIM1 -> m (DVol DIM1)
softMaxBackward label output _ = return $ traverse output id gradientAt
      where
        gradientAt f s@(Z :. i) = gradient (f s) i
        gradient outA target = -(bool2Double indicator - outA)
            where
              indicator = label == target
              bool2Double x = if x then 1.0 else 0.0

-- |'InnerLayer' represents an inner layer of a neural network that
-- can accept backpropagation input from higher layers
class (Layer a sh sh', Shape sh, Shape sh') => InnerLayer a sh sh' | a -> sh, a -> sh' where
    innerBackward :: Monad m => a -> Vol sh' -> Vol sh -> m (DVol sh)

-- |'FullyConnectedLayer' represents a fully-connected input layer
data FullyConnectedLayer sh = FullyConnectedLayer {
      _weights :: Vol (sh :. Int),
      _bias    :: Vol DIM1
    }

instance (Shape sh) => Layer (FullyConnectedLayer sh) sh DIM1 where
    forward = fcForward

instance (Shape sh) => InnerLayer (FullyConnectedLayer sh) sh DIM1 where
    innerBackward = fcBackward

fcForward :: (Shape sh, Monad m)
          => FullyConnectedLayer sh -> Vol sh -> m (DVol DIM1)
fcForward (FullyConnectedLayer w b) input =
    return $ traverse w toNumFilters f
        where
          toNumFilters (_ :. i) = Z :. i
          f _ (Z :. i) = bias + dotProduct weights input
              where
                bias = toUnboxed b V.! i
                weights = computeUnboxedS $ slice w (Any :. (i :: Int))

fcBackward :: (Monad m)
           => FullyConnectedLayer sh -> Vol DIM1 -> Vol sh -> m (DVol sh)
fcBackward = undefined

dotProduct :: (Num a, V.Unbox a) => Array U sh a -> Array U sh a -> a
dotProduct l r = prod (toUnboxed l) (toUnboxed r)
    where
      prod lv rv = V.sum $ V.zipWith (*) lv rv


-- |The 'Forward' function represents a single forward pass through a layer.
type Forward m sh sh' = (Vol sh -> WriterT [V.Vector Double] m (DVol sh'))

-- |'>->' composes two forward activation functions
(>->) :: (Monad m, Shape sh, Shape sh', Shape sh'')
        => Forward m sh sh' -> Forward m sh' sh'' -> Forward m sh sh''
(f >-> g) input = do
  intermediate <- f input
  unboxed <- computeP intermediate
  tell [toUnboxed unboxed]
  g unboxed

-- |'net1' constructs a single-layer fully connected perceptron with
-- softmax output.
net1
  :: (Monad m, InnerLayer a sh DIM1, TopLayer a1) =>
     a -> a1 -> Forward m sh DIM1
net1 bottom top = forward bottom >-> forward top

-- |'net1' constructs a two-layer fully connected MLP with
-- softmax output.
net2
  :: (Monad m, InnerLayer a sh sh', InnerLayer a1 sh' DIM1,
      TopLayer a2) =>
     a -> a1 -> a2 -> Forward m sh DIM1
net2 bottom middle top = forward bottom >-> net1 middle top

-- |'withActivations' computes the output activation, along with the
-- intermediate activations
withActivations :: Forward m sh sh' -> Vol sh -> m (DVol sh', [V.Vector Double])
withActivations f input = runWriterT (f input)

-- |'newFC' constructs a new fully connected layer
newFC :: Shape sh => sh -> Int -> FullyConnectedLayer sh
newFC sh numFilters = FullyConnectedLayer {
                        _weights=randomishDoubleArray (sh :. (numFilters :: Int)) 0 1.0 1,
                        _bias=randomishDoubleArray (Z :. (numFilters :: Int)) 0 1.0 1
                      }

-- |'FlowNetwork' builds a network of the form
--
-- @
--  Input Layer              Output Softmax
--     +--+
--     |  |   Inner Layers    +--+   +--+
--     |  |                   |  |   |  |
--     |  |   +-+   +-+  +-+  |  |   |  |
--     |  +---+ +---+ +--+ +--+  +--->  |
--     |  |   +-+   +-+  +-+  |  |   |  |
--     |  |                   |  |   |  |
--     |  |                   +--+   +--+
--     +--+
-- @
flowNetwork :: (Monad m, Shape sh) => sh -> Int -> Int -> Int -> Forward m sh DIM1
flowNetwork inputShape numHiddenLayers numHiddenNodes numClasses =
    inputLayer >-> innerLayers >-> preTopLayer >-> topLayer
        where
          flatInner layers = P.foldl1 (>->) (P.fmap forward layers)
          innerLayer = newFC (Z :. numHiddenNodes) numHiddenNodes
          innerLayers = flatInner $ P.fmap (const innerLayer)
                                           [1..numHiddenLayers]
          inputLayer = forward $ newFC inputShape numHiddenNodes
          preTopLayer = forward $ newFC (Z :. numHiddenNodes) numClasses
          topLayer = forward SoftMaxLayer
