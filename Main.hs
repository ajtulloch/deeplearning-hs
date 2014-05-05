{-# LANGUAGE TypeOperators #-}
module Main where

import           Data.Array.Repa
import           DeepLearning.ConvNet
import           DeepLearning.Util

main :: IO ()
main = do
  (pvol, acts) <- withActivations (testNet testShape 2) (testInput testShape)
  print (computeS pvol :: Vol DIM1)
  print acts
