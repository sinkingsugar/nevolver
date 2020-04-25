(import "../build/libcbnevolver.dll")

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
           (* 10 npixels))
                                        ; use it
   (Sequence .final :Types Type.Float)
   (Repeat
    (-->
     .subject
     (Convolve 2)
     (ImageToFloats)
     (Nevolver.Predict .mlp) >> .final)
    npixels)
                                        ; write result
   .final
   (Flatten)
   (FloatsToImage 128 128 3)
   (WritePNG "result.png")

   (Nevolver.SaveModel .mlp)
   (WriteFile "mlp-saved.nn")
   ))

(schedule Root sgded)

(run Root)
(def sgded nil)
(prn "Done 1")

(def replay
  (Chain
   "replay"
                                        ; load subject
   (LoadImage "cat_original.png")
   (StripAlpha) &> .subject
                                        ; load pre-trained model
   (ReadFile "mlp-saved.nn") (ExpectBytes)
   (Nevolver.LoadModel .loaded-mlp)
                                        ; infer the model on subject
   (Sequence .final :Types Type.Float)
   (Repeat
    (-->
     .subject
     (Convolve 2)
     (ImageToFloats)
     (Nevolver.Predict .loaded-mlp) >> .final)
    npixels)
                                        ; write results
   .final
   (Flatten)
   (FloatsToImage 128 128 3)
   (WritePNG "result2.png")))

(schedule Root replay)

(run Root)
(def replay nil)
(def Root nil)
(prn "Done 2")

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
