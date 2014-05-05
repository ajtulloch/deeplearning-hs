
{-# LANGUAGE TypeOperators #-}
module Main where

import           CPU.ConvNet
import           Data.Array.Repa
import           Data.Array.Repa.Algorithms.Randomish
import           Data.Array.Repa.Arbitrary
import           Data.Monoid
import qualified Data.Vector.Unboxed                  as V
import           Test.Framework
import           Test.Framework.Providers.QuickCheck2
import           Test.QuickCheck

testShape :: (Z :. Int) :. Int
testShape = Z :. (3 :: Int) :. (3 :: Int)

testInput :: Shape sh => sh -> Array U sh Double
testInput sh = randomishDoubleArray sh 0 1.0 1

testNet :: (Monad m, Shape sh) => sh -> Int -> Forward m sh DIM1
testNet sh numFilters = net1 testFC testSM
    where
      testFC = newFC sh numFilters
      testSM = SoftMaxLayer

demo :: IO ()
demo = do
  (pvol, acts) <- withActivations (testNet testShape 2) (testInput testShape)
  print (computeS pvol :: Vol DIM1)
  print acts


genOneLayer :: (Shape sh) => sh -> Gen (Int, Vol sh)
genOneLayer sh = do
  a <- choose (1, 10)
  b <- arbitraryUShaped sh
  return (a, b)

testFilter :: (Shape sh) => (Int, Vol sh) -> Bool
testFilter (numFilters, input) = and invariants
    where
      [(outAP, [innerA])] = withActivations (testNet sh numFilters) (testInput sh)
      outA = computeS outAP :: Vol DIM1
      sh = extent input
      invariants = [
       (length . toList) outA == numFilters,
       V.length innerA == numFilters]


prop_singleLayer :: Property
prop_singleLayer = forAll (genOneLayer testShape) testFilter

tests :: [Test]
tests = [testProperty "singleLayer" prop_singleLayer]


main :: IO ()
main = defaultMainWithOpts tests mempty
