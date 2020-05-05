(import "../build/nevolver.dll")

(def Root (Node))

(def mlp
  (Chain
   "mlp"
   (Mutant
    (Nevolver.MLP .mlp
                  :Inputs 2
                  :Hidden 4
                  :Outputs 1))
   (Const [0.0 1.0])
   (Nevolver.Predict .mlp) >== .result))

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
  :Looped
  (Once (--> 0 >= .times))
                                        ; evolve 20 times
  .times (Math.Inc) > .times
  (When (IsMore 20) (Stop))
                                        ; train
  (Evolve mlp fitness :Population 100 :Coroutines 10)
  (Log) (Take 1) >= .chain
  (WriteFile "best.nn")
  (ChainRunner .chain)
  (Get .result :Global true :Default [-1.0]) (Log "best prediction")
  ))

(run Root 0.1)
(prn "Done training")

(def testLoad
  (Chain
   "loadNN"
   (ReadFile "best.nn")
   (ExpectChain) >= .chain
   (ChainRunner .chain)
   (Get .result :Global true :Default [-1.0])
   (Log)
   ))

(schedule Root testLoad)
(run Root 0.1)
(prn "Done load test")
