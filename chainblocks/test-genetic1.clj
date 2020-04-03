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
          (Nevolver.Activate .mlp)))

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
     :Population 10000
     :Coroutines 100)
    (Log)
    (Ref "best"))
   15)
  (Get "best")
  (Log)
  ))

(run Root 0.1)
(prn "Done")
