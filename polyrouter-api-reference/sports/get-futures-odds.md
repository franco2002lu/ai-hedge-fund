# Get futures odds

> Retrieve real-time odds for a specific futures market (championships, divisions, conferences) from multiple prediction market platforms using future ID.

## OpenAPI

````yaml openapi.json get /futures/{future_id}
paths:
  path: /futures/{future_id}
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
      path:
        future_id:
          schema:
            - type: string
              required: true
              description: 'Future identifier (format: {league}_{future_type}_{season})'
              example: nfl_superbowl_2025
      query:
        platform:
          schema:
            - type: enum<string>
              enum:
                - polymarket
                - kalshi
              required: false
              description: 'Filter by platforms (comma-separated): polymarket, kalshi'
        odds_format:
          schema:
            - type: enum<string>
              enum:
                - american
                - decimal
                - probability
              required: false
              description: Odds format
              default: american
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              futures:
                allOf:
                  - type: array
                    items:
                      $ref: '#/components/schemas/Future'
              pagination:
                allOf:
                  - $ref: '#/components/schemas/Pagination'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              futures:
                - id: nfl_superbowl_2025
                  future_name: Super Bowl LX Winner
                  league: nfl
                  season: 2025
                  future_type: championship
                  category: team
                  markets:
                    - platform: <string>
                      market_id: <string>
                      outcomes:
                        - team_polyrouter_id: <string>
                          outcome_name: <string>
                          odds:
                            american: <string>
                            decimal: 123
                            implied_probability: 123
                          volume_24h: 123
                          last_trade_price: 123
                          metadata: {}
                  metadata: {}
              pagination:
                total: 123
                limit: 123
                offset: 123
                has_more: true
                next_offset: 123
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Futures markets retrieved successfully
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
  deprecated: false
  type: path
components:
  schemas:
    Pagination:
      type: object
      required:
        - total
        - limit
        - offset
        - has_more
        - next_offset
      properties:
        total:
          type: number
          description: Total number of results
        limit:
          type: number
          description: Number of results per page
        offset:
          type: number
          description: Current page offset
        has_more:
          type: boolean
          description: Whether more pages exist
        next_offset:
          type: number
          description: Offset for next page
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
    Future:
      type: object
      required:
        - id
        - future_name
        - league
        - season
        - future_type
        - category
        - markets
      properties:
        id:
          type: string
          description: Future identifier
          example: nfl_superbowl_2025
        future_name:
          type: string
          description: Full future name
          example: Super Bowl LX Winner
        league:
          type: string
          description: League identifier
          example: nfl
        season:
          type: integer
          description: Season year
          example: 2025
        future_type:
          type: string
          description: Future type
          enum:
            - championship
            - division
            - conference
          example: championship
        category:
          type: string
          description: Category
          example: team
        markets:
          type: array
          items:
            type: object
            properties:
              platform:
                type: string
                description: Platform name
              market_id:
                type: string
                description: Platform-specific market identifier
              outcomes:
                type: array
                items:
                  type: object
                  properties:
                    team_polyrouter_id:
                      type: string
                      nullable: true
                      description: PolyRouter team ID
                    outcome_name:
                      type: string
                      description: Team name
                    odds:
                      type: object
                      properties:
                        american:
                          type: string
                          description: American odds
                        decimal:
                          type: number
                          description: Decimal odds
                        implied_probability:
                          type: number
                          description: Implied probability
                    volume_24h:
                      type: number
                      description: 24-hour trading volume
                    last_trade_price:
                      type: number
                      description: Last trade price (0-1)
                    metadata:
                      type: object
                      description: Platform-specific metadata
        metadata:
          type: object
          description: Future metadata and platform-specific info

````