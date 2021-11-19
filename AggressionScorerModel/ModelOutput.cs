using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace AggressionScorerModel
{
   public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}
