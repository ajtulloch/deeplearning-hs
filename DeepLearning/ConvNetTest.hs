{-# LANGUAGE TypeOperators #-}
module Main where

import           Data.Array.Repa
import           Data.Array.Repa.Arbitrary
import           Data.Monoid
import qualified Data.Vector.Unboxed                  as V
import           DeepLearning.ConvNet
import           DeepLearning.Util
import           Test.Framework
import           Test.Framework.Providers.QuickCheck2
import           Test.QuickCheck


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
