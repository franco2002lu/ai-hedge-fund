# Get price history

> Retrieve historical price data using platform-specific market identifiers. This endpoint uses platform-native IDs and fetches data directly from platform APIs for improved performance. Currently, only Polymarket, Kalshi, and Limitless are supported.

## OpenAPI

````yaml openapi.json get /price-history
paths:
  path: /price-history
  method: get
  servers:
    - url: https://api.polyrouter.io/functions/v1
      description: Production server
  request:
    security:
      - title: ApiKeyAuth
        parameters:
          query: {}
          header:
            X-API-Key:
              type: apiKey
              description: API key for authentication
          cookie: {}
    parameters:
      path: {}
      query:
        market_ids:
          schema:
            - type: string
              required: true
              description: Comma-separated list of platform-specific market IDs
        start_ts:
          schema:
            - type: integer
              required: true
              description: Start timestamp (Unix timestamp) - Required for data retrieval
              default: 1759333191
        end_ts:
          schema:
            - type: integer
              required: true
              description: End timestamp (Unix timestamp) - Required for data retrieval
              default: 1760283591
        interval:
          schema:
            - type: enum<string>
              enum:
                - 1m
                - 5m
                - 1h
                - 4h
                - 1d
              required: true
              description: Time interval - Required
              default: 1d
        limit:
          schema:
            - type: integer
              required: false
              description: Maximum number of data points per market (1-5000)
              maximum: 5000
              minimum: 1
              default: 10
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              data:
                allOf:
                  - type: array
                    items:
                      $ref: '#/components/schemas/PriceHistoryPointV2'
              meta:
                allOf:
                  - type: object
                    properties:
                      total_points:
                        type: number
                        description: Total number of data points returned
                      platforms:
                        type: array
                        items:
                          type: string
                        description: List of platforms included in response
                      time_range:
                        type: object
                        properties:
                          start:
                            type: number
                          end:
                            type: number
                      interval:
                        type: string
                      request_time:
                        type: number
                      cache_hit:
                        type: boolean
                      data_freshness:
                        type: string
                        format: date-time
        examples:
          example:
            value:
              data:
                - timestamp: 123
                  price:
                    close: 123
                    open: 123
                    high: 123
                    low: 123
                  volume: 123
                  openInterest: 123
                  bidAsk:
                    bid:
                      close: 123
                      open: 123
                      high: 123
                      low: 123
                    ask:
                      close: 123
                      open: 123
                      high: 123
                      low: 123
                  platform: kalshi
                  marketId: <string>
                  outcomeId: <string>
              meta:
                total_points: 123
                platforms:
                  - <string>
                time_range:
                  start: 123
                  end: 123
                interval: <string>
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Price history retrieved successfully
    '400':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Validation error response
    '401':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
        description: Authentication error response
    '404':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Resource not found error response
    '500':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Internal server error response
    '503':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Platform unavailable
  deprecated: false
  type: path
components:
  schemas:
    Meta:
      type: object
      required:
        - request_time
        - cache_hit
        - data_freshness
      properties:
        request_time:
          type: number
          description: Request processing time (ms)
        cache_hit:
          type: boolean
          description: Whether response was served from cache
        data_freshness:
          type: string
          format: date-time
          description: Data freshness timestamp
    Error:
      type: object
      required:
        - code
        - message
        - timestamp
      properties:
        code:
          type: string
          description: Error code
          example: VALIDATION_ERROR
        message:
          type: string
          description: Error message
        details:
          type: object
          description: Additional error details
        timestamp:
          type: string
          format: date-time
          description: Error timestamp
    PriceData:
      type: object
      required:
        - close
      properties:
        close:
          type: number
          description: Closing price (always available)
        open:
          type: number
          description: Opening price (Kalshi only)
        high:
          type: number
          description: Highest price (Kalshi only)
        low:
          type: number
          description: Lowest price (Kalshi only)
    BidAskData:
      type: object
      properties:
        bid:
          $ref: '#/components/schemas/OHLCData'
        ask:
          $ref: '#/components/schemas/OHLCData'
    OHLCData:
      type: object
      properties:
        close:
          type: number
        open:
          type: number
        high:
          type: number
        low:
          type: number
    PriceHistoryPointV2:
      type: object
      required:
        - timestamp
        - price
        - platform
        - marketId
      properties:
        timestamp:
          type: number
          description: Unix timestamp of the data point
        price:
          $ref: '#/components/schemas/PriceData'
        volume:
          type: number
          nullable: true
          description: Trading volume
        openInterest:
          type: number
          nullable: true
          description: Open interest
        bidAsk:
          $ref: '#/components/schemas/BidAskData'
        platform:
          type: string
          enum:
            - kalshi
            - polymarket
            - limitless
          description: Platform identifier
        marketId:
          type: string
          description: Platform-specific market identifier
        outcomeId:
          type: string
          description: Outcome identifier (yes, no, etc.)

````