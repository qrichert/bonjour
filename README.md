# bonjour

> [!CAUTION]
>
> Work in progress, first usable version hasn't landed yet!

Experimental crate to extract first names with a confidence level.

Expected output may be something like this:

```json
{
  "input": "Quentin Richert",
  "first_name": "Quentin",
  "confidence": 0.95
}
```

The idea is that is also "detects", or at lease significantly reduces
confidences in company names, for instance:

```json
// The company marker 'SAS' significantly reduces confidence.
{
  "input": "Quentin Richert SAS",
  "first_name": "Quentin",
  "confidence": 0.1
}
```

Up to no detection at all:

```json
{
  "input": "Les Motards d'Alsace",
  "first_name": null,
  "confidence": 0.0
}
```

## License

The source code is available under the [0BSD license](LICENSE).

Datasets distributed with or used to build this project are compiled
from publicly available information. They are not covered by the 0BSD
license; their contents remain subject to any applicable rights and
source terms.
