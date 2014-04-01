{-# LANGUAGE FlexibleContexts       #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE RankNTypes             #-}
{-# LANGUAGE TypeOperators          #-}

module CPU.ConvNet where

import           Control.Monad
import           Data.Array.Repa
import qualified Data.Vector.Unboxed as V
import           Prelude             hiding (map, zipWith)

-- ** Helper Types

type Vol sh = Array U sh Double
type DVol sh = Array D sh Double

type Label = Int

-- **  Top Layers
class TopLayer a where
    topForward :: (Monad m) => a -> Vol DIM1 -> m (DVol DIM1)
    topBackward :: (Monad m) => a -> Label -> Vol DIM1 -> Vol DIM1 -> m (DVol DIM1)

-- |SoftMaxLayer
data SoftMaxLayer = SoftMaxLayer

instance TopLayer SoftMaxLayer where
    topForward _ = softMaxForward
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

-- ** Inner Layers
class (Shape sh, Shape sh') => InnerLayer a sh sh' | a -> sh where
    innerForward :: Monad m => a -> Vol sh -> m (DVol sh')
    innerBackward :: Monad m => a -> Vol sh' -> Vol sh -> m (DVol sh)

-- |FullyConnectedLayer
data FullyConnectedLayer sh = FullyConnectedLayer {
      _weights :: Array U (sh :. Int) Double,
      _bias    :: Array U DIM1 Double
    }

instance (Shape sh) => InnerLayer (FullyConnectedLayer sh) sh DIM1 where
    innerForward = fcForward
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


-- ** Composing Layers
type Forward m sh sh' = (Vol sh -> m (DVol sh'))

compose :: (Monad m, Shape sh, Shape sh', Shape sh'')
        => Forward m sh sh' -> Forward m sh' sh'' -> Forward m sh sh''
compose f g input = do
  intermediate <- f input
  unboxed <- computeP intermediate
  g unboxed

oneLayerSoftMax
  :: (Monad m, InnerLayer a sh DIM1, TopLayer a1) =>
     a -> a1 -> Forward m sh DIM1
oneLayerSoftMax bottom top = compose (innerForward bottom) (topForward top)

twoLayerSoftMax
  :: (Monad m, InnerLayer a sh sh', InnerLayer a1 sh' DIM1,
      TopLayer a2) =>
     a -> a1 -> a2 -> Forward m sh DIM1
twoLayerSoftMax bottom middle top = compose (innerForward bottom)
                                    (oneLayerSoftMax middle top)
