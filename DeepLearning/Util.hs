{-# LANGUAGE TypeOperators #-}
module DeepLearning.Util where

import           Data.Array.Repa
import           Data.Array.Repa.Algorithms.Randomish
import           Data.Array.Repa.Arbitrary
import           Data.Monoid
import qualified Data.Vector.Unboxed                  as V
import           DeepLearning.ConvNet

testShape :: (Z :. Int) :. Int
testShape = Z :. (3 :: Int) :. (3 :: Int)

testInput :: Shape sh => sh -> Array U sh Double
testInput sh = randomishDoubleArray sh 0 1.0 1

testNet :: (Monad m, Shape sh) => sh -> Int -> Forward m sh DIM1
testNet sh numFilters = net1 testFC testSM
    where
      testFC = newFC sh numFilters
      testSM = SoftMaxLayer
