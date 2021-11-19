using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.ML;

namespace AggressionScorerModel
{
   public class AggressionScore
   {
       private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool;

       public AggressionScore(PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool)
       {
           _predictionEnginePool = predictionEnginePool;
       }

       public ModelOutput Predict(string input) => 
       _predictionEnginePool.Predict(new ModelInput()
       {
           Comment = input
       });
   }
}
