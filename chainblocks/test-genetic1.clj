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
          (Nevolver.Activate .mlp)
          (Log "Activation")))

(def fitness
  (Chain
   "fitness"
   (Math.Subtract 1.0)
   (ToFloat)
   (Math.Abs)
   (Math.Multiply -1.0)
   (Log "Fitness")))

(schedule
 Root
 (Chain
  "test"
  (Repeat
   (-->
    (Evolve
     mlp
     fitness
     :Population 16
     :Coroutines 2
     :Crossover 1.0
     :Mutation 1.0)
    (Log)
    (Ref "best"))
   15)
  (Get "best")
  (Log)
  ))

(run Root 0.1)
(prn "Done")
