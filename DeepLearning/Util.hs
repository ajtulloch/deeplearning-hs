{-# LANGUAGE TypeOperators #-}

{-|
Module      : DeepLearning.Util
Description : Deep Learning
Copyright   : (c) Andrew Tulloch, 2014
License     : GPL-3
Maintainer  : andrew+cabal@tullo.ch
Stability   : experimental
Portability : POSIX
-}
module DeepLearning.Util where

import           Data.Array.Repa
import           Data.Array.Repa.Algorithms.Randomish
import           DeepLearning.ConvNet

-- |Sample 3x3 matrix used for demonstrations and tests
testShape :: (Z :. Int) :. Int
testShape = Z :. (3 :: Int) :. (3 :: Int)

-- |Random 3x3 matrix
testInput :: Shape sh => sh -> Array U sh Double
testInput sh = randomishDoubleArray sh 0 1.0 1

-- |Random single-layer network
testNet :: (Monad m, Shape sh) => sh -> Int -> Forward m sh DIM1
testNet sh numFilters = net1 testFC testSM
    where
      testFC = newFC sh numFilters
      testSM = SoftMaxLayer
