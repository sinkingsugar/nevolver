(import "../build/libcbnevolver.dll")

(def Root (Node))

(def mlp (Chain
          "mlp"
          (Mutant
           (Nevolver.MLP .mlp
                         :Inputs 2
                         :Hidden 4
                         :Outputs 1))
          (Const [0.0 1.0])
          (Nevolver.Predict .mlp)
          (Log)))

(def fitness
  (Chain
   "fitness"
   (Math.Subtract 1.0)
   (ToFloat)
   (Math.Abs)
   (Math.Multiply -1.0)))

(schedule
 Root
 (Chain
  "test"
  (Repeat
   (-->
    (Evolve
     mlp
     fitness
     :Population 1000
     :Coroutines 100)
    (Log) &> .best)
   2)
  .best
  (Log)
  (Take 1) >= .chain
  (WriteFile "best.nn")
  (ChainRunner .chain)
  ))

(run Root 0.1)
(prn "Done")

(def testLoad
  (Chain
   "loadNN"
   (ReadFile "best.nn")
   (ExpectChain) >= .chain
   (ChainRunner .chain)
   ))

(schedule Root testLoad)
(run Root 0.1)
(prn "Done")
