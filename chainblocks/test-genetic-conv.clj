(import "../build2/nevolver.dll")
(import "../build-lnx/nevolver.so")

(def Root (Node))

(def npixels (* 128 128))
(def eras 1500)

(def job
  (Chain
   "job"
                                        ; load resources
   (LoadImage "cat_texture.png")
   (StripAlpha) &> .filtered
   (LoadImage "cat.png")
   (StripAlpha) &> .unfiltered
                                        ; build the network
   (Mutant
    (Nevolver.MLP .mlp
                  :Inputs (* 3 (* 3 3))
                  :Hidden 8
                  :Outputs 3))
                                        ; train it
   (Sequence .diffs :Types [[Type.Float]])
   (Repeat (-->
            .unfiltered
            (Convolve 2 2)
            (ImageToFloats)
            (Nevolver.Predict .mlp)
            >= .predicted

            .filtered
            (Convolve 1 2)
            (ImageToFloats)
            (Math.Subtract .predicted)
            (Math.Pow 2.0)
            >> .diffs)
           npixels)
                                        ; save model if required
   (Get .dumpModel :Default false)
   (When (Is true) (-->
                    (Nevolver.SaveModel .mlp)
                    (WriteFile "best-conv.nn")))
                                        ; prepare fitness
   .diffs
   (Reduce (Math.Add .$0))
   (Math.Mean)))

(def fitness
  (Chain
   "fitness"
   (Math.Multiply -1.0)))

(schedule
 Root
 (Chain
  "evolution"
  :Looped
  (Once (--> 0 >= .times))
                                        ; evolve 20 times
  .times (Math.Inc) > .times
  (When (IsMore eras) (Stop))
                                        ; train
  (Evolve job fitness :Population 100 :Threads 10 :Coroutines 2)
                                        ; print, store
  (Log) (Take 1) >= .chain
  true >== .dumpModel
  (ChainRunner .chain)
  (Msg "DONE")))

(run Root)

(def replay
  (Chain
   "replay"
                                        ; load subject
   (LoadImage "cat_original.png")
   (StripAlpha) &> .subject
                                        ; load pre-trained model
   (ReadFile "best-conv.nn") (ExpectBytes)
   (Nevolver.LoadModel .loaded)
                                        ; infer the model on subject
   (Sequence .final :Types Type.Float)
   (Repeat
    (-->
     .subject
     (Convolve 2)
     (ImageToFloats)
     (Nevolver.Predict .loaded) >> .final)
    npixels)
                                        ; write results
   .final
   (Flatten)
   (FloatsToImage 128 128 3)
   (WritePNG "result-evolved.png")))

(schedule Root replay)

(run Root)
