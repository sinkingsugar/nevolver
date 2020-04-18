(import "../build2/libcbnevolver.dll")

(def Root (Node))

(def npixels (* 128 128))

(def sgded
  (Chain
   "sgd"
                                        ; load resources
   (LoadImage "cat_texture.png")
   (StripAlpha) &> .filtered
   (LoadImage "cat.png")
   (StripAlpha) &> .unfiltered
   (LoadImage "cat_original.png")
   (StripAlpha) &> .subject
                                        ; build the network
   (Nevolver.MLP .mlp
                 :Inputs (* 3 (* 3 3))
                 :Hidden 8
                 :Outputs 3)
                                        ; train it
   (Repeat (-->
            .unfiltered
            (Convolve 2 2)
            (ImageToFloats)
            (Nevolver.Activate .mlp)

            .filtered
            (Convolve 1 2)
            (ImageToFloats)
            (Nevolver.Propagate .mlp))
           (* 100 npixels))
                                        ; use it
   (Repeat
    (-->
     .subject
     (Convolve 2)
     (ImageToFloats)
     (Nevolver.Predict .mlp) >>! .final)
    npixels)
                                        ; write result
   .final
   (Flatten)
   (FloatsToImage 128 128 3)
   (WritePNG "result.png")))

(schedule Root sgded)

;; (def mlp (Chain
;;           "mlp"
;;           (Mutant
;;            (Nevolver.MLP .mlp
;;                          :Inputs 2
;;                          :Hidden 4
;;                          :Outputs 1))
;;           (Const [0.0 1.0])
;;           (Nevolver.Activate .mlp)))

;; (def fitness
;;   (Chain
;;    "fitness"
;;    (Math.Subtract 1.0)
;;    (ToFloat)
;;    (Math.Abs)
;;    (Math.Multiply -1.0)))

;; (schedule
;;  Root
;;  (Chain
;;   "test"
;;   (Repeat
;;    (-->
;;     (Evolve
;;      mlp
;;      fitness
;;      :Population 10000
;;      :Coroutines 100)
;;     (Log)
;;     (Ref "best"))
;;    15)
;;   (Get "best")
;;   (Log)
;;   ))

(run Root 0.01)
(prn "Done")
