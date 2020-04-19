(import "../build2/libcbnevolver.dll")

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
          (Set .result :Global true)))

(def fitness
  (Chain
   "fitness"
   (Math.Subtract 2.0)
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
     :Population 100
     :Coroutines 10)
    (Log) &> .best)
   20)
  .best
  (Log)
  (Take 1) >= .chain
  (WriteFile "best.nn")
  (ChainRunner .chain)
  (Get .result :Default [-1.0])
  (Log)
  ))

(run Root 0.1)
(prn "Done")

(def testLoad
  (Chain
   "loadNN"
   (ReadFile "best.nn")
   (ExpectChain) >= .chain
   (ChainRunner .chain)
   (Get .result :Default [-1.0])
   (Log)
   ))

(schedule Root testLoad)
(run Root 0.1)
(prn "Done")
