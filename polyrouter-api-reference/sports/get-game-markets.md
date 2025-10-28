# Get game markets

> Retrieve detailed betting markets for a specific game from multiple platforms (Polymarket, Kalshi, ProphetX, Novig) using PolyRouter game ID. Keep in mind, Polymarket and Kalshi often make game markets available the day of, while ProphetX, Novig, and SXBET don't currently offer historical data.

## OpenAPI

````yaml openapi.json get /games/{game_id}
paths:
  path: /games/{game_id}
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
        game_id:
          schema:
            - type: string
              required: true
              description: 'PolyRouter game ID (format: {AwayTeam}v{HomeTeam}{YYYYMMDD})'
              example: PITvBAL20251012
      query:
        platform:
          schema:
            - type: enum<string>
              enum:
                - polymarket
                - kalshi
                - prophetx
                - novig
                - sxbet
              required: false
              description: >-
                Filter by platforms (comma-separated): polymarket, kalshi,
                prophetx, novig, sxbet
        market_type:
          schema:
            - type: enum<string>
              enum:
                - moneyline
                - spread
                - total
                - prop
              required: false
              description: >-
                Filter by market type: moneyline, spread, total, prop (player
                props)
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              games:
                allOf:
                  - type: array
                    items:
                      $ref: '#/components/schemas/GameMarkets'
              pagination:
                allOf:
                  - $ref: '#/components/schemas/Pagination'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              games:
                - id: DETvKC20251012
                  title: <string>
                  teams:
                    - <string>
                  sport: <string>
                  league: <string>
                  description: <string>
                  scheduled_at: '2023-11-07T05:31:56Z'
                  status: scheduled
                  markets:
                    - platform: <string>
                      event_id: <string>
                      outcomes:
                        - name: <string>
                          price: 123
                          volume: 123
                          status: <string>
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
        description: Game markets retrieved successfully
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
    GameMarkets:
      type: object
      required:
        - id
        - title
        - teams
        - sport
        - league
        - markets
      properties:
        id:
          type: string
          description: PolyRouter game ID
          example: DETvKC20251012
        title:
          type: string
          description: Game title with team names
        teams:
          type: array
          items:
            type: string
          description: Array of PolyRouter team IDs
        sport:
          type: string
          description: Sport type
        league:
          type: string
          description: League identifier
        description:
          type: string
          description: Game description
        scheduled_at:
          type: string
          format: date-time
          description: Game start time (ISO 8601)
        status:
          type: string
          enum:
            - scheduled
            - live
            - final
          description: Game status
        markets:
          type: array
          items:
            type: object
            properties:
              platform:
                type: string
                description: Platform name
              event_id:
                type: string
                description: Platform event identifier
              outcomes:
                type: array
                items:
                  type: object
                  properties:
                    name:
                      type: string
                      description: Outcome name
                    price:
                      type: number
                      description: Current price (0-1)
                    volume:
                      type: number
                      description: Trading volume
                    status:
                      type: string
                      description: Outcome status
              metadata:
                type: object
                description: Platform-specific metadata
        metadata:
          type: object
          description: Additional game metadata

````