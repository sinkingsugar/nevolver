(import "../build/libcbnevolver.dll")

(def Root (Node))

(def mlp (Chain
          "mlp"
          :Looped
          (Nevolver.MLP .mlp
                        :Inputs 2
                        :Hidden 4
                        :Outputs 1)
          (Const [0.0 1.0])
          (Nevolver.Activate .mlp)
          (Log "prediction")
          (Const [1.0])
          (Nevolver.Propagate .mlp)
          (Log "error")
          ))

(schedule Root mlp)

(run Root 0.01 100)
