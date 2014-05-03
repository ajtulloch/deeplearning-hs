{-# LANGUAGE FlexibleContexts       #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE RankNTypes             #-}
{-# LANGUAGE ScopedTypeVariables    #-}
{-# LANGUAGE TypeOperators          #-}

module CPU.ConvNet where

import           Data.Array.Accelerate
import           Data.Array.Accelerate.Interpreter
import           Prelude                           as P hiding (map, sum,
                                                         zipWith)
-- ** Helper Types

type Vol sh = Array sh Double

type Label = Int

-- **  Top Layers
class TopLayer a where
    topForward :: a -> Acc (Vol DIM1) -> Acc (Vol DIM1)
    topBackward :: a -> Label -> Acc (Vol DIM1) -> Acc (Vol DIM1) -> Acc (Vol DIM1)

-- |SoftMaxLayer
data SoftMaxLayer = SoftMaxLayer

instance TopLayer SoftMaxLayer where
    topForward _ = softMaxForward
    topBackward _ = softMaxBackward

softMaxForward :: Acc (Vol DIM1) -> Acc (Vol DIM1)
softMaxForward input =  map division exponentials
   where
     division :: Exp Double -> Exp Double
     division x = x / the sumE
     exponentials :: Acc (Vol DIM1)
     exponentials = exponentiate input

     sumE :: Acc (Scalar Double)
     sumE = fold (+) 0.0 exponentials

     maxA :: Acc (Scalar Double)
     maxA = fold max 0.0 exponentials

     exponentiate = map (\a -> exp (a - the maxA))

softMaxBackward :: Label -> Acc (Vol DIM1) -> Acc (Vol DIM1) -> Acc (Vol DIM1)
softMaxBackward label output _ = zipWith gradient output (enumFromN (shape output) 0)
      where
        gradient :: Exp Double -> Exp Int -> Exp Double
        gradient outA target = -(bool2Double indicator - outA)
            where
              indicator = lift label == target
              bool2Double x = if x then 1.0 else 0.0

-- ** Inner Layers
class (Shape sh, Shape sh') => InnerLayer a sh sh' | a -> sh, a -> sh' where
    innerForward :: a -> Acc (Vol sh) -> Acc (Vol sh')
    innerBackward :: a -> Acc (Vol sh') -> Acc (Vol sh) -> Acc (Vol sh)

-- |FullyConnectedLayer
data FullyConnectedLayer sh = FullyConnectedLayer {
      _weights :: [Vol sh],
      _bias    :: [Double]
    }

instance (Shape sh, Slice sh) => InnerLayer (FullyConnectedLayer sh) sh DIM1 where
    innerForward = fcForward
    innerBackward = fcBackward

fcForward :: Shape sh => FullyConnectedLayer sh -> Acc (Vol sh) -> Acc (Vol DIM1)
fcForward (FullyConnectedLayer w b) input = use $ fromList (Z :. length b) results
    where
      results :: [Exp Double]
      results = map the (map run filterOps)
      filterOps :: [Acc (Scalar Double)]
      filterOps = (map runFilter (P.zip w b)) :: [Acc (Scalar Double)]
      runFilter :: (Vol sh, Double) -> Acc (Scalar Double)
      runFilter (weights, bias) = unit $ (lift bias) + dotProduct (use weights) input


-- fwd ::  (Shape sh) => Acc (sh :. Int) -> Acc (Vol DIM1) -> Acc (Vol sh)-> Acc (Vol DIM1)
-- fwd weights bias input = result
--     where
--       (Z :. numFilters) = unlift (shape bias) :: (Z :. Exp Int)
--       result = generate (lift $ Z :. numFilters) f :: (Acc (Vol DIM1))
--       f :: Exp DIM1 -> Exp Double
--       f idx = b + dotProduct w input
--           where
--             (Z :. i) = unlift idx :: (Z :. Exp Int)
--             b = bias ! idx :: Exp Double
--             w = slice weights (lift $ (sh :. i))
--       -- arrRepl = replicate (lift $ any :. numFilters) (use w)
--       -- use $ fromList (index1 10) $ zipWith f w b
--     -- where
--     --   f :: Vol sh -> Double -> Exp Double
--     --   f = undefined

fcBackward :: Shape sh => FullyConnectedLayer sh -> Acc (Vol DIM1) -> Acc (Vol sh) -> Acc (Vol sh)
fcBackward = undefined

dotProduct :: (Shape sh) => Acc (Vol sh) -> Acc (Vol sh) -> Exp Double
dotProduct l r = the $ sum $ zipWith (*) l r


-- ** Composing Layers
type Forward sh sh' = (Acc (Vol sh) -> Acc (Vol sh'))

oneLayerSoftMax
  :: (InnerLayer a sh DIM1, TopLayer a1) =>
     a -> a1 -> Acc (Vol sh) -> Acc (Vol DIM1)
oneLayerSoftMax bottom top = innerForward bottom >-> topForward top

twoLayerSoftMax
  :: (InnerLayer a sh sh', InnerLayer a1 sh' DIM1, TopLayer a2) =>
     a -> a1 -> a2 -> Acc (Vol sh) -> Acc (Vol DIM1)
twoLayerSoftMax bottom middle top = innerForward bottom >-> oneLayerSoftMax middle top
